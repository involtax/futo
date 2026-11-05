package org.futo.voiceinput.shared.gigaam

import android.content.res.AssetManager
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineNemoEncDecCtcModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig

data class DecodedSpeech(val text: String, val tokens: List<String>, val timestamps: List<Float>) {
    companion object {
        /** E2E SentencePiece tokens may contain both a word boundary and a word fragment. */
        fun fromPieces(text: String, pieces: List<String>, times: List<Float>): DecodedSpeech {
            require(pieces.size == times.size)
            val tokens = mutableListOf<String>()
            val timestamps = mutableListOf<Float>()
            pieces.forEachIndexed { index, piece ->
                Regex("\\s+|\\S+").findAll(piece.replace('▁', ' ')).forEach {
                    tokens.add(it.value)
                    timestamps.add(times[index])
                }
            }
            return DecodedSpeech(text.trim(), tokens, timestamps)
        }
    }
}

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
            return DecodedSpeech.fromPieces(result.text, result.tokens.toList(), result.timestamps.toList())
        } finally {
            stream.release()
        }
    }

    fun close() = recognizer.release()
}
