package ro.mta.landmarkrecognitionapp;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.pm.ResolveInfo;
import android.net.Uri;
import android.os.Bundle;
import android.widget.ImageButton;
import android.widget.Toast;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;
import androidx.viewpager2.widget.ViewPager2;

import com.google.android.material.button.MaterialButton;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class ViewPhotoRecognizedActivity extends AppCompatActivity {
    private String recognizedImagesDirLocation;
    private ImageViewPager imageViewPager;
    private ViewPager2 viewPager;
    ArrayList<String> recognizedImagesList;
    private ImageButton deleteButton;
    private ImageButton shareImage;
    private MaterialButton detailsImageButton;
    private LandmarkRecognitionDatabase landmarkRecognitionDatabase;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_view_photo_recognized);
        recognizedImagesDirLocation = getFilesDir().getPath() + "/Recognized Images";
        landmarkRecognitionDatabase = LandmarkRecognitionDatabase.getInstance(this);
        viewPager = findViewById(R.id.photo_view_pager_rec);
        File recognizedImagesDir = new File(recognizedImagesDirLocation);
        File[] files = recognizedImagesDir.listFiles();
        recognizedImagesList = new ArrayList<>();
        assert files != null;
        for (File file : files) {
            recognizedImagesList.add(file.getAbsolutePath());
        }
        if (recognizedImagesList.size() > 1)
            recognizedImagesList.sort(String::compareTo);
        imageViewPager = new ImageViewPager(recognizedImagesList, this);
        viewPager.setAdapter(imageViewPager);

        Intent intent = getIntent();
        viewPager.setCurrentItem(intent.getIntExtra("position", imageViewPager.getItemCount() - 1));

        deleteButton = findViewById(R.id.delete_button_rec);
        deleteButton.setOnClickListener(view -> {
            AlertDialog.Builder builder = new AlertDialog.Builder(this);
            builder.setCancelable(true);
            builder.setTitle("Delete Image");
            builder.setMessage("Do you want to delete this image?");
            builder.setPositiveButton(android.R.string.yes,
                    (dialog, which) -> {
                        int pos = viewPager.getCurrentItem();
                        if (pos == 0) {
                            if (recognizedImagesList.size() == 1) {
                                deleteFile(pos);
                                finish();
                            } else {
                                viewPager.setCurrentItem(pos + 1);
                                deleteFile(pos);
                            }
                        } else {
                            viewPager.setCurrentItem(pos - 1);
                            deleteFile(pos);
                        }
                    });
            builder.setNegativeButton(android.R.string.no, (dialog, which) -> {
            }).setIcon(android.R.drawable.ic_dialog_alert);
            AlertDialog dialog = builder.create();
            dialog.show();
        });

        shareImage = findViewById(R.id.share_button_rec);
        shareImage.setOnClickListener(view -> {
            Uri uriToImage = FileProvider.getUriForFile(this, "ro.mta.landmarkrecognitionapp.provider", new File(recognizedImagesList.get(viewPager.getCurrentItem())));
            Intent shareIntent = new Intent();
            shareIntent.setAction(Intent.ACTION_SEND);
            shareIntent.putExtra(Intent.EXTRA_STREAM, uriToImage);
            shareIntent.setFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            shareIntent.setType("image/jpeg");
            Intent chooser = Intent.createChooser(shareIntent, "Share File");
            List<ResolveInfo> resInfoList = this.getPackageManager().queryIntentActivities(chooser, PackageManager.MATCH_DEFAULT_ONLY);
            for (ResolveInfo resolveInfo : resInfoList) {
                String packageName = resolveInfo.activityInfo.packageName;
                this.grantUriPermission(packageName, uriToImage, Intent.FLAG_GRANT_WRITE_URI_PERMISSION | Intent.FLAG_GRANT_READ_URI_PERMISSION);
            }
            startActivity(chooser);
        });

        detailsImageButton = findViewById(R.id.details_button_rec);
        detailsImageButton.setOnClickListener(view -> {
            int pos = viewPager.getCurrentItem();
            RecognizedImages image = landmarkRecognitionDatabase.recognizedImagesDao().getImageByPath(recognizedImagesList.get(pos));
            File file = new File(recognizedImagesList.get(pos));
            Toast.makeText(this, "Image " + file.getName() +
                    "\nLandmark Name: " + image.getLandmarkName() +
                    "\nLandmark Country: " + image.getCountry() +
                    "\nLandmark Locality: " + image.getLocality() +
                    "\nLandmark Latitude: " + image.getLatitude() +
                    "\nLandmark Longitude: " + image.getLongitude(), Toast.LENGTH_SHORT).show();
        });
    }

    private void deleteFile(int pos) {
        RecognizedImages imageToBeDeleted = landmarkRecognitionDatabase.recognizedImagesDao().getImageByPath(recognizedImagesList.get(pos));
        landmarkRecognitionDatabase.recognizedImagesDao().deleteRecognizedImages(imageToBeDeleted);
        new File(recognizedImagesList.get(pos)).getAbsoluteFile().delete();
        recognizedImagesList.remove(pos);
        imageViewPager.notifyItemRemoved(pos);
    }
}