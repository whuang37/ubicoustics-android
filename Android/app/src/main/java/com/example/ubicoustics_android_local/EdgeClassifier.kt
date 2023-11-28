package com.example.ubicoustics_android_local

import android.content.Context
import android.util.Log
import androidx.compose.runtime.MutableState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import org.tensorflow.lite.task.audio.classifier.AudioClassifier
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.Timer
import kotlin.concurrent.scheduleAtFixedRate

class EdgeClassifier(private val context: Context) {
    companion object {
        private const val MODEL_PATH = "example_model_quantized_meta.tflite"
        private const val PROBABILITY_THRESHOLD: Float = 0.3f
        private const val POLLING_RATE_MILLIS = 100L
        private const val WRITE_TO_FILE = true
    }

    var dataFile: File? = null

     public fun classifyByEdge(loggingText: MutableState<String>) {
         if (WRITE_TO_FILE) {
             dataFile = createDataFile()
         }

        val classifier = AudioClassifier.createFromFile(this.context, MODEL_PATH)
        // create audio recorder
        val tensor = classifier.createInputTensorAudio()

        // showing audio recorder specs
        val format = classifier.requiredTensorAudioFormat
        val recorderSpecs = "Number Of Channels: ${format.channels}\n" +
                "Sample Rate: ${format.sampleRate}"
        Log.i("Edge Info", recorderSpecs)

        // creating
        val record = classifier.createAudioRecord()
        record.startRecording()

        Timer().scheduleAtFixedRate(1, POLLING_RATE_MILLIS) {
            val startTime = System.currentTimeMillis()

            tensor.load(record)
            val output = classifier.classify(tensor)

            val endTime = System.currentTimeMillis()
            val delay = endTime - startTime

            val topPrediction = output[0].categories.sortedBy{ -it.score }[0]
            val logString = edgeDataToString(startTime, topPrediction.label, topPrediction.score, delay)
            writeDataToFile(dataFile, logString)

            val filteredModelOutput = output[0].categories.filter {
                it.score > PROBABILITY_THRESHOLD
            }

            val outputStr =
                filteredModelOutput.sortedBy { -it.score }
                    .joinToString(separator = "\n") { "${it.label} -> ${it.score} " }

            if (outputStr.isNotEmpty()) {
                loggingText.value = outputStr
            }
        }
    }

    private fun createDataFile(): File {
        val dataFile = File(context.getExternalFilesDir(null), System.currentTimeMillis().toString() + ".csv")
//        val dataFile = File(context.filesDir, System.currentTimeMillis().toString() + ".csv")
        val absolutePath = dataFile.absolutePath

        Log.i("Edge Logging", absolutePath)

        writeDataToFile(dataFile, "timestamp,label,probability,delay")
        return dataFile
    }

    private fun edgeDataToString(timestamp: Long, label: String, probability: Float, delay: Long): String {
        return "$timestamp,$label,$probability,$delay"
    }

    private fun writeDataToFile(dataFile: File?, data: String) {
        if (dataFile == null) {
            return
        }

        GlobalScope.launch(Dispatchers.IO) {
            // Use a coroutine in the IO dispatcher to perform file I/O in the background
            try {
                // Open the file in append mode
                val fileOutputStream = FileOutputStream(dataFile, true)

                // Write the data to the file
                fileOutputStream.write(data.toByteArray())
                fileOutputStream.write("\n".toByteArray()) // Add a newline for better readability

                // Close the file output stream
                fileOutputStream.close()

            } catch (e: IOException) {
                // Handle exceptions (e.g., file not found, permission issues)
                e.printStackTrace()
            }
        }

    }
}