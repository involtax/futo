package org.futo.voiceinput.shared.gigaam

import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.async
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.futo.voiceinput.shared.ggml.InferenceCancelledException
import org.futo.voiceinput.shared.ggml.inferenceContext
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/** Runs the packaged Kotlin decoder against the matching Linux JNI build on real Russian audio. */
class GigaAmInferenceTest {
    private fun words(text: String) = text.lowercase().replace('ё', 'е')
        .replace(Regex("[^\\p{L}\\p{N}]+"), " ").trim()
    private fun withModel(test: (GigaAmDecoder, FloatArray) -> Unit) {
        val path = System.getProperty("gigaam.modelDir")
        assumeTrue("Set GIGAAM_TEST_MODEL_DIR and SHERPA_JNI_DIR for native inference tests", path != null)
        val dir = File(path!!)
        val bytes = File(dir, "test_wavs/example.wav").readBytes()
        val wav = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        assertEquals("RIFF", String(bytes.copyOfRange(0, 4)))
        var offset = 12
        var samples: FloatArray? = null
        while (offset + 8 <= bytes.size) {
            val name = String(bytes.copyOfRange(offset, offset + 4))
            val size = wav.getInt(offset + 4)
            if (name == "fmt ") {
                assertEquals(1, wav.getShort(offset + 8).toInt())
                assertEquals(1, wav.getShort(offset + 10).toInt())
                assertEquals(16000, wav.getInt(offset + 12))
                assertEquals(16, wav.getShort(offset + 22).toInt())
            }
            if (name == "data") samples = FloatArray(size / 2) {
                wav.getShort(offset + 8 + it * 2).toFloat() / 32768f
            }
            offset += 8 + size + (size % 2)
        }
        val model = GigaAmDecoder(File(dir, "model.int8.onnx").path, File(dir, "tokens.txt").path)
        try { test(model, samples!!) } finally { model.close() }
    }

    @Test fun transcribesWholeRussianRecording() = withModel { model, audio ->
        val result = model.decode(audio)
        assertTrue(result.text, words(result.text).startsWith("ничьих не требуя похвал"))
        assertTrue(result.text, words(result.text).endsWith("у лукоморья дуб зеленый"))
        assertTrue("E2E capitalization missing: ${result.text}", result.text.first().isUpperCase())
        assertTrue("E2E punctuation missing: ${result.text}", result.text.any { it in ",.!?:;" })
        assertEquals(result.tokens.size, result.timestamps.size)
        assertTrue(result.tokens.any { it.isBlank() })
        println("GigaAM whole-recording: ${result.text}")
    }

    @Test fun cancelledRequestCannotPublishAndNextSessionStillWorks() = withModel { decoder, audio ->
        runBlocking {
            val model = GigaAmModel(decoder)
            val entered = CountDownLatch(1)
            val release = CountDownLatch(1)
            val blocker = launch(inferenceContext) {
                entered.countDown()
                check(release.await(5, TimeUnit.SECONDS))
            }
            assertTrue(entered.await(5, TimeUnit.SECONDS))
            val oldRequest = async(start = CoroutineStart.UNDISPATCHED) { runCatching { model.decode(audio) } }
            model.cancel()
            release.countDown()
            blocker.join()
            assertTrue(oldRequest.await().exceptionOrNull() is InferenceCancelledException)
            assertTrue(words(model.decode(audio).text).endsWith("у лукоморья дуб зеленый"))
        }
    }

    private fun live(model: GigaAmDecoder, audio: FloatArray): Pair<String, Int> {
        val buffer = LiveDictationBuffer()
        var previewsBeforeStop = 0
        for (start in audio.indices step 1600) {
            val block = audio.copyOfRange(start, minOf(start + 1600, audio.size))
            buffer.append(block, block.any { abs(it) > 0.015f })
            buffer.snapshot(false)?.let {
                buffer.accept(it, model.decode(it.samples))
                if (buffer.text.isNotBlank()) previewsBeforeStop++
            }
        }
        while (true) {
            val s = buffer.snapshot(true) ?: break
            buffer.accept(s, model.decode(s.samples))
        }
        return buffer.text to previewsBeforeStop
    }

    @Test fun producesRussianWordsBeforeStopAndFinalizesTail() = withModel { model, audio ->
        val (text, partials) = live(model, audio)
        assertTrue("Expected multiple live previews, got $partials", partials >= 3)
        assertTrue(text, words(text).contains("ничьих не требуя похвал"))
        assertTrue(text, words(text).endsWith("у лукоморья дуб зеленый"))
        println("GigaAM live ($partials previews before stop): $text")
    }

    @Test fun longDictationRollsWindowsAndKeepsRepeatedSpeech() = withModel { model, audio ->
        val repeated = audio + audio + audio
        val (text, partials) = live(model, repeated)
        assertTrue(partials >= 15)
        val phrase = "ничьих не требуя похвал"
        assertEquals(text, 3, Regex(phrase).findAll(words(text)).count())
        assertTrue(text, words(text).endsWith("у лукоморья дуб зеленый"))
        assertFalse("Repeated word at a window boundary: $text", words(text).contains("у у лукоморья"))
        println("GigaAM long live ($partials previews): $text")
    }
}
