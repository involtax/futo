package org.futo.voiceinput.shared.gigaam

import org.junit.Assert.*
import org.junit.Test

class LiveDictationBufferTest {
    private fun result(text: String) = DecodedSpeech(text, emptyList(), emptyList())

    @Test fun sentencePiecesExposeWordBoundariesAndKeepPunctuation() {
        val decoded = DecodedSpeech.fromPieces("Привет, мир!",
            listOf("▁При", "вет", ",", "▁мир", "!"), listOf(0f, 0.2f, 0.3f, 0.5f, 0.6f))
        assertEquals("Привет, мир!", decoded.tokens.joinToString("").trim())
        assertEquals(listOf(" ", "При", "вет", ",", " ", "мир", "!"), decoded.tokens)
        assertEquals(decoded.tokens.size, decoded.timestamps.size)
        assertEquals(0.5f, decoded.timestamps[4])
    }

    @Test fun silenceNeverInvokesRecognition() {
        val b = LiveDictationBuffer()
        repeat(100) { b.append(FloatArray(1600), false) }
        assertNull(b.snapshot(false))
        assertNull(b.snapshot(true))
        assertEquals("", b.text)
    }

    @Test fun previewsReplaceTheTailInsteadOfAppendingDuplicates() {
        val b = LiveDictationBuffer()
        b.append(FloatArray(16000), true)
        b.accept(b.snapshot(false)!!, result("я хочу билет"))
        b.append(FloatArray(16000), true)
        b.accept(b.snapshot(false)!!, result("я хочу билеты на завтра"))
        assertEquals("я хочу билеты на завтра", b.text)
        assertNull(b.snapshot(false))
    }

    @Test fun pauseCommitsOnlyTheSnapshotAndRetainsNewAudio() {
        val b = LiveDictationBuffer()
        b.append(FloatArray(16000) { 1f }, true)
        b.append(FloatArray(11200), false)
        val first = b.snapshot(false)!!
        assertTrue(first.finalize)
        b.append(FloatArray(16000) { 2f }, true) // microphone continues during inference
        b.accept(first, result("первая фраза"))
        val second = b.snapshot(true)!!
        assertEquals(16000, second.samples.size)
        assertTrue(second.samples.all { it == 2f })
        b.accept(second, result("вторая фраза"))
        assertEquals("первая фраза вторая фраза", b.text)
        assertNull(b.snapshot(true))
    }

    @Test fun rolloverUsesWordTimestampAndDoesNotLoseAudio() {
        val b = LiveDictationBuffer()
        b.append(FloatArray(16000 * 14) { it.toFloat() }, true)
        val s = b.snapshot(false)!!
        assertTrue(s.rollover)
        b.accept(s, DecodedSpeech("один два", listOf("о", "д", "и", "н", " ", "д", "в", "а"),
            listOf(0f, 1f, 2f, 3f, 8f, 9f, 10f, 11f)))
        assertEquals("один два", b.text)
        val tail = b.snapshot(true)!!
        assertEquals(16000 * 5 + 8000, tail.samples.size)
        assertEquals((16000 * 8 + 8000).toFloat(), tail.samples.first())
        b.accept(tail, result("два три"))
        assertEquals("один два три", b.text)
    }

    @Test fun stopIncludesShortLastWord() {
        val b = LiveDictationBuffer()
        b.append(FloatArray(6400), true)
        assertNull(b.snapshot(false))
        b.accept(b.snapshot(true)!!, result("да"))
        assertEquals("да", b.text)
    }

    @Test fun overlapRemovalPreservesWordsThatWereActuallyRepeated() {
        val b = LiveDictationBuffer()
        b.append(FloatArray(16000 * 14), true)
        b.accept(b.snapshot(false)!!, DecodedSpeech("один да да", listOf("один", " ", "да", " ", "да"),
            listOf(0f, 8f, 8.5f, 9f, 10f)))
        b.accept(b.snapshot(true)!!, DecodedSpeech("да да два", listOf("да", " ", "да", " ", "два"),
            listOf(0.3f, 1f, 2f, 3f, 4f)))
        assertEquals("один да да два", b.text)
    }

    @Test fun overlapMatchingIgnoresE2eCaseAndPunctuationWithoutChangingNewText() {
        val b = LiveDictationBuffer()
        b.append(FloatArray(16000 * 14), true)
        b.accept(b.snapshot(false)!!, DecodedSpeech("Один, Да да", listOf("Один,", " ", "Да", " ", "да"),
            listOf(0f, 8f, 8.5f, 9f, 10f)))
        b.accept(b.snapshot(true)!!, DecodedSpeech("да Да два.", listOf("да", " ", "Да", " ", "два."),
            listOf(0.3f, 1f, 2f, 3f, 4f)))
        assertEquals("Один, Да Да два.", b.text)
    }

    @Test fun emptyFinalResultRemovesAnIncorrectPreview() {
        val b = LiveDictationBuffer()
        b.append(FloatArray(16000), true)
        b.accept(b.snapshot(false)!!, result("ошибка"))
        b.accept(b.snapshot(true)!!, result(""))
        assertEquals("", b.text)
    }
}
