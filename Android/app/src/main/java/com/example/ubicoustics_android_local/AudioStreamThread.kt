
import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.compose.runtime.MutableState
import kotlinx.coroutines.*
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress
import java.nio.ByteBuffer

class AudioStreamThread(private val loggingText: MutableState<String>) : Thread() {

    companion object {
        private const val SAMPLE_RATE = 16000
        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
        private val BUFFER_SIZE = 32000
//            AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
        private const val PORT = 12345 // Change this to your desired UDP port
        private const val POLLING_RATE_MILLIS = 100L // Adjust the polling rate in milliseconds
        private const val SERVER_ADDRESS: String = "0.0.0.0" // Change this to your server's IP address
    }

    private var audioRecord: AudioRecord? = null
    private var job: Job? = null

    private var prevDelay = 0L

    override fun run() {
        try {
            // Start audio recording in a separate coroutine
            job = CoroutineScope(Dispatchers.Default).launch {
                startAudioRecording()
            }
            Log.i("audio stream", "STREAM STARTED")

            // Initialize UDP socket and server address
            val socket = DatagramSocket()
            val serverInetAddress = InetAddress.getByName(SERVER_ADDRESS)

            val buffer = ByteArray(BUFFER_SIZE)
            while (!isInterrupted) {
                // compute the overall delay between recording and sending of the previous packet
                // for logging purposes
                val startTimestamp = System.currentTimeMillis()
                val bytesRead = audioRecord?.read(buffer, 0, buffer.size) ?: 0

                val delayBuffer = ByteBuffer.allocate(8)
                delayBuffer.putLong(prevDelay)
                val timestampBytes = delayBuffer.array()


                // Combine timestamp and audio data
                val combinedData = ByteArray(timestampBytes.size + bytesRead)
                System.arraycopy(timestampBytes, 0, combinedData, 0, timestampBytes.size)
                System.arraycopy(buffer, 0, combinedData, timestampBytes.size, bytesRead)

                val packet = DatagramPacket(combinedData, combinedData.size, serverInetAddress, PORT)
                socket.send(packet)

                val endTimestamp = System.currentTimeMillis()
                prevDelay = endTimestamp - startTimestamp // timing delay of the previous send

                loggingText.value = "Sent " + timestampBytes.size.toString() + " bytes of audio in " + prevDelay.toString() + " ms of delay"
                sleep(POLLING_RATE_MILLIS)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            audioRecord?.release()
        }
    }

    @SuppressLint("MissingPermission")
    private fun startAudioRecording() {
        audioRecord =
            AudioRecord(MediaRecorder.AudioSource.MIC, SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT, BUFFER_SIZE)
        audioRecord?.startRecording()
    }

    fun stopStreaming() {
        job?.cancel()
        interrupt()
    }
}
