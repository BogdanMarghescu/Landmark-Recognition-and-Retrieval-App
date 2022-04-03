package ro.mta.landmarkrecognitionapp;

import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.View;
import android.widget.ImageButton;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.Preview;
import androidx.camera.extensions.HdrImageCaptureExtender;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class CameraActivity extends AppCompatActivity {
    private Executor executor = Executors.newSingleThreadExecutor();
    private int REQUEST_CODE_PERMISSIONS = 1001;
    private final String[] REQUIRED_PERMISSIONS = new String[]{"android.permission.CAMERA", "android.permission.WRITE_EXTERNAL_STORAGE"};
    private PreviewView previewView;
    private ImageButton cameraCaptureButton;
    private ImageButton flashlightSetButton;
    private int cameraSelectorPosition = CameraSelector.LENS_FACING_BACK;
    private int flashMode = ImageCapture.FLASH_MODE_AUTO;
    private String unrecognizedImagesDirLocation;
    private String recognizedImagesDirLocation;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);
        unrecognizedImagesDirLocation = getFilesDir().getPath() + "/Unrecognized Images";
        recognizedImagesDirLocation = getFilesDir().getPath() + "/Recognized Images";
        File unrecognizedImagesDir = new File(unrecognizedImagesDirLocation);
        File recognizedImagesDir = new File(recognizedImagesDirLocation);
        if (!unrecognizedImagesDir.exists())
            unrecognizedImagesDir.mkdirs();
        if (!recognizedImagesDir.exists())
            recognizedImagesDir.mkdirs();
        previewView = findViewById(R.id.previewView);
        cameraCaptureButton = findViewById(R.id.camera_capture_button);
        ImageButton cameraSwitchButton = findViewById(R.id.camera_switch_button);
        cameraSwitchButton.setOnClickListener(view -> {
            cameraSelectorPosition = 1 - cameraSelectorPosition;
            startCamera();
        });
        flashlightSetButton = findViewById(R.id.flashlight_set);
        flashlightSetButton.setOnClickListener(view -> {
            flashMode = (flashMode + 1) % 3;
            if (flashMode == ImageCapture.FLASH_MODE_AUTO)
                flashlightSetButton.setImageResource(R.drawable.baseline_flash_auto_24);
            else if (flashMode == ImageCapture.FLASH_MODE_ON)
                flashlightSetButton.setImageResource(R.drawable.baseline_flash_on_24);
            else if (flashMode == ImageCapture.FLASH_MODE_OFF)
                flashlightSetButton.setImageResource(R.drawable.baseline_flash_off_24);
        });
        if (allPermissionsGranted()) {
            startCamera();
        } else {
            int REQUEST_CODE_PERMISSIONS = 1001;
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
        }
    }

    private void startCamera() {
        final ListenableFuture<ProcessCameraProvider> cameraProviderListenableFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderListenableFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderListenableFuture.get();
                bindPreview(cameraProvider);
            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    void bindPreview(@NonNull ProcessCameraProvider cameraProvider) {
        if (cameraSelectorPosition == CameraSelector.LENS_FACING_FRONT) {
            flashlightSetButton.setVisibility(View.GONE);
        } else {
            flashlightSetButton.setVisibility(View.VISIBLE);
        }
        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());
        CameraSelector cameraSelector = new CameraSelector.Builder().requireLensFacing(cameraSelectorPosition).build();
        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder().build();
        ImageCapture.Builder builder = new ImageCapture.Builder();
        HdrImageCaptureExtender hdrImageCaptureExtender = HdrImageCaptureExtender.create(builder);
        if (hdrImageCaptureExtender.isExtensionAvailable(cameraSelector)) {
            hdrImageCaptureExtender.enableExtension(cameraSelector);
        }
        final ImageCapture imageCapture = builder.setTargetRotation(this.getWindowManager().getDefaultDisplay().getRotation()).build();
        cameraProvider.unbindAll();
        Camera camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis, imageCapture);
        cameraCaptureButton.setOnClickListener(view -> {
            SimpleDateFormat dateFormat = new SimpleDateFormat("yyyyMMddHHmmss", Locale.US);
            File file = new File(unrecognizedImagesDirLocation, dateFormat.format(new Date()) + ".jpg");
            ImageCapture.OutputFileOptions outputFileOptions = new ImageCapture.OutputFileOptions.Builder(file).build();
            imageCapture.setFlashMode(flashMode);
            imageCapture.takePicture(outputFileOptions, executor, new ImageCapture.OnImageSavedCallback() {
                @Override
                public void onImageSaved(@NonNull ImageCapture.OutputFileResults outputFileResults) {
                    new Handler(Looper.getMainLooper()).post(() -> Toast.makeText(CameraActivity.this, "Image Saved Succesfully to " + file.getAbsolutePath(), Toast.LENGTH_SHORT).show());
                }

                @Override
                public void onError(@NonNull ImageCaptureException exception) {
                    exception.printStackTrace();
                }
            });
        });
    }

    private boolean allPermissionsGranted() {
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera();
            } else {
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show();
                this.finish();
            }
        }
    }
}