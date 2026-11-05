package org.futo.voiceinput.shared.types

import org.futo.voiceinput.shared.ggml.DecodingMode

interface SpeechModel {
    suspend fun infer(samples: FloatArray, prompt: String, languages: Array<String>,
        bailLanguages: Array<String>, decodingMode: DecodingMode,
        suppressNonSpeechTokens: Boolean, partialResultCallback: (String) -> Unit): String
    fun cancel()
    suspend fun close()
}
