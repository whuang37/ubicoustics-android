package com.example.ubicoustics_android_local

import AudioStreamThread
import android.Manifest.permission.ACCESS_NETWORK_STATE
import android.Manifest.permission.INTERNET
import android.Manifest.permission.READ_EXTERNAL_STORAGE
import android.Manifest.permission.RECORD_AUDIO
import android.Manifest.permission.WRITE_EXTERNAL_STORAGE
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.app.ActivityCompat
import com.example.ubicoustics_android_local.ui.theme.UbicousticsAndroidLocalTheme

class MainActivity : ComponentActivity() {

    private val loggingText = mutableStateOf("")
    private var audioStreamThread: AudioStreamThread? = null
    private var edgeClassifier: EdgeClassifier? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            UbicousticsAndroidLocalTheme {
                // A surface container using the 'background' color from the theme
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
//                    Greeting("Android")
                    LoggingOutput()
                    InitButtons()
                }
            }
        }
        checkPermission()
    }

    private fun checkPermission() {
        // Function to check and request permission.
        val permissionsCode = 42
        val permissions = arrayOf(RECORD_AUDIO, READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE, INTERNET, ACCESS_NETWORK_STATE)
        // Requesting the permission
        ActivityCompat.requestPermissions(this@MainActivity, permissions, permissionsCode)
    }

    fun startEdgeClassifier() {
        edgeClassifier = EdgeClassifier(this)
        edgeClassifier?.classifyByEdge(loggingText)
    }

    fun startAudioStreaming() {
        audioStreamThread = AudioStreamThread(loggingText)
        audioStreamThread?.start()
        loggingText.value = "STARTED STREAMING"
    }

    @Composable
    fun InitButtons() {
        var showButtons by remember { mutableStateOf(true) }
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically
        ) {
            if (showButtons) {
                Button( onClick = {
                        showButtons = false;
                        startEdgeClassifier()
                    },
                    modifier = Modifier.size(width= 140.dp, height= 40.dp)
                ) {
                    Text(
                        text = "Edge",
                        style = TextStyle(fontSize=15.sp)
                    )
                }
                Button( onClick = {
                    showButtons = false;
                    startAudioStreaming()
                    },
                    modifier = Modifier.size(width= 140.dp, height= 40.dp)
                ) {
                    Text(
                        text = "Stream",
                        style = TextStyle(fontSize=15.sp)
                    )
                }
            }
        }
    }

    @Composable
    fun LoggingOutput() {
        val text by loggingText
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(text,
                textAlign = TextAlign.Center)
        }
    }
}
