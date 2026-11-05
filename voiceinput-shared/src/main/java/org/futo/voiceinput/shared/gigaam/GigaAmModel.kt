package org.futo.voiceinput.shared.gigaam

import android.content.Context
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import org.futo.voiceinput.shared.R
import org.futo.voiceinput.shared.ggml.DecodingMode
import org.futo.voiceinput.shared.ggml.InferenceCancelledException
import org.futo.voiceinput.shared.ggml.InvalidModelException
import org.futo.voiceinput.shared.ggml.inferenceContext
import org.futo.voiceinput.shared.types.ModelLoader
import org.futo.voiceinput.shared.types.SpeechModel
import java.util.concurrent.atomic.AtomicInteger
import kotlin.coroutines.coroutineContext

object BuiltInGigaAm : ModelLoader {
    override val name = R.string.gigaam_russian_name
    override fun exists(context: Context) = true
    override fun getRequiredDownloadList(context: Context) = emptyList<String>()
    override fun key(context: Context): Any = "gigaam-v3-e2e-ctc-int8-d5fea8df"
    override fun loadModel(context: Context): SpeechModel = try {
        GigaAmModel(GigaAmDecoder("gigaam/model.int8.onnx", "gigaam/tokens.txt", context.assets))
    } catch (e: Exception) {
        throw InvalidModelException()
    }
}

class GigaAmModel(private val decoder: GigaAmDecoder) : SpeechModel {
    private val generation = AtomicInteger()

    suspend fun decode(samples: FloatArray): DecodedSpeech {
        val request = generation.get()
        return withContext(inferenceContext) {
            coroutineContext.ensureActive()
            if (generation.get() != request) throw InferenceCancelledException()
            val result = decoder.decode(samples)
            coroutineContext.ensureActive()
            if (generation.get() != request) throw InferenceCancelledException()
            result
        }
    }

    override suspend fun infer(samples: FloatArray, prompt: String, languages: Array<String>,
        bailLanguages: Array<String>, decodingMode: DecodingMode,
        suppressNonSpeechTokens: Boolean, partialResultCallback: (String) -> Unit): String {
        val text = decode(samples).text
        partialResultCallback(text)
        return text
    }

    override fun cancel() { generation.incrementAndGet() }
    override suspend fun close() = withContext(inferenceContext) { decoder.close() }
}
