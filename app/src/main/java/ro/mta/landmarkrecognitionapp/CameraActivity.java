package ro.mta.landmarkrecognitionapp;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.ImageButton;
import android.widget.Toast;

import androidx.annotation.DrawableRes;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
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
import androidx.core.os.ConfigurationCompat;

import com.bumptech.glide.Glide;
import com.bumptech.glide.request.RequestOptions;
import com.google.common.util.concurrent.ListenableFuture;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Date;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class CameraActivity extends AppCompatActivity {
    private Executor executor = Executors.newSingleThreadExecutor();
    private final int REQUEST_CODE_PERMISSIONS = 1001;
    private final String[] REQUIRED_PERMISSIONS = new String[]{"android.permission.CAMERA", "android.permission.WRITE_EXTERNAL_STORAGE"};
    private PreviewView previewView;
    private ImageButton cameraCaptureButton;
    private ImageButton flashlightSetButton;
    private ImageButton photoViewButton;
    private int cameraSelectorPosition = CameraSelector.LENS_FACING_BACK;
    private int flashMode = ImageCapture.FLASH_MODE_AUTO;
    private String unrecognizedImagesDirLocation;
    private LandmarkRecognitionDatabase landmarkRecognitionDatabase;
    @DrawableRes
    int[] flashIconSet = {R.drawable.baseline_flash_auto_24, R.drawable.baseline_flash_on_24, R.drawable.baseline_flash_off_24};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);
        Window w = getWindow(); // in Activity's onCreate() for instance
        w.setFlags(WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS, WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS);
        unrecognizedImagesDirLocation = getFilesDir().getPath() + "/Unrecognized Images";
        landmarkRecognitionDatabase = LandmarkRecognitionDatabase.getInstance(this);
        previewView = findViewById(R.id.previewView);

        cameraCaptureButton = findViewById(R.id.camera_capture_button);
        ImageButton cameraSwitchButton = findViewById(R.id.camera_switch_button);
        cameraSwitchButton.setOnClickListener(view -> {
            cameraSelectorPosition ^= 1;
            startCamera();
        });

        flashlightSetButton = findViewById(R.id.flashlight_set);
        flashlightSetButton.setImageResource(flashIconSet[flashMode]);
        flashlightSetButton.setOnClickListener(view -> {
            flashMode = (flashMode + 1) % 3;
            flashlightSetButton.setImageResource(flashIconSet[flashMode]);
        });

        photoViewButton = findViewById(R.id.photo_view_button);
        String lastImagePath = getLastImagePath(unrecognizedImagesDirLocation);
        if (lastImagePath != null) setGalleryThumbnail(lastImagePath);
        photoViewButton.setOnClickListener(view -> {
            String lastImagePath1 = getLastImagePath(unrecognizedImagesDirLocation);
            if (lastImagePath1 != null)
                startActivity(new Intent(getApplicationContext(), ViewPhotoUnrecognizedActivity.class));
            else
                Toast.makeText(CameraActivity.this, "There are no images!", Toast.LENGTH_SHORT).show();
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
            SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss", ConfigurationCompat.getLocales(getResources().getConfiguration()).get(0));
            File file = new File(unrecognizedImagesDirLocation, dateFormat.format(new Date()) + ".jpg");
            ImageCapture.OutputFileOptions outputFileOptions = new ImageCapture.OutputFileOptions.Builder(file).build();
            imageCapture.setFlashMode(flashMode);
            imageCapture.takePicture(outputFileOptions, executor, new ImageCapture.OnImageSavedCallback() {
                @Override
                public void onImageSaved(@NonNull ImageCapture.OutputFileResults outputFileResults) {
                    new Handler(Looper.getMainLooper()).post(() -> {
                        UnrecognizedImages newImage = new UnrecognizedImages(file.getAbsolutePath());
                        if (landmarkRecognitionDatabase.unrecognizedImagesDao().getCountByPath(file.getAbsolutePath()) == 0)
                            landmarkRecognitionDatabase.unrecognizedImagesDao().insertUnrecognizedImages(newImage);
                        Toast.makeText(CameraActivity.this, "Image Saved Succesfully as " + file.getName(), Toast.LENGTH_SHORT).show();
                        setGalleryThumbnail(file.getAbsolutePath());
                    });
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

    private void setGalleryThumbnail(String imagePath) {
        int padding = getResources().getDimensionPixelOffset(R.dimen.stroke_small);
        photoViewButton.setPadding(padding, padding, padding, padding);
        Glide.with(photoViewButton).load(imagePath).apply(RequestOptions.circleCropTransform()).into(photoViewButton);
    }

    @Nullable
    private String getLastImagePath(String imageFolderPath) {
        File unrecognizedImagesDir = new File(imageFolderPath);
        File[] unrecognizedImagesList = unrecognizedImagesDir.listFiles();
        if (unrecognizedImagesList != null && unrecognizedImagesList.length > 1)
            Arrays.sort(unrecognizedImagesList, Comparator.comparing(File::getName));
        assert unrecognizedImagesList != null;
        if (unrecognizedImagesList.length > 0)
            return unrecognizedImagesList[unrecognizedImagesList.length - 1].getAbsolutePath();
        return null;
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

    @Override
    public void onResume() {
        super.onResume();
        String lastImagePath = getLastImagePath(unrecognizedImagesDirLocation);
        if (lastImagePath != null)
            setGalleryThumbnail(lastImagePath);
        else {
            int padding = getResources().getDimensionPixelOffset(R.dimen.spacing_large);
            photoViewButton.setPadding(padding, padding, padding, padding);
            photoViewButton.setImageResource(R.drawable.ic_photo);
        }
    }
}