package org.futo.voiceinput.shared.gigaam

import android.content.res.AssetManager
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineNemoEncDecCtcModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig

data class DecodedSpeech(val text: String, val tokens: List<String>, val timestamps: List<Float>)

/** The same decoder is used for whole recordings and live audio windows. */
class GigaAmDecoder(model: String, tokens: String, assets: AssetManager? = null) {
    private val recognizer = OfflineRecognizer(assets, OfflineRecognizerConfig(
        featConfig = FeatureConfig(sampleRate = 16000, featureDim = 64),
        modelConfig = OfflineModelConfig(
            nemo = OfflineNemoEncDecCtcModelConfig(model = model),
            tokens = tokens, numThreads = 2, provider = "cpu"
        )
    ))

    fun decode(samples: FloatArray): DecodedSpeech {
        val stream = recognizer.createStream()
        try {
            stream.acceptWaveform(samples, 16000)
            recognizer.decode(stream)
            val result = recognizer.getResult(stream)
            return DecodedSpeech(result.text.trim(), result.tokens.toList(), result.timestamps.toList())
        } finally {
            stream.release()
        }
    }

    fun close() = recognizer.release()
}
