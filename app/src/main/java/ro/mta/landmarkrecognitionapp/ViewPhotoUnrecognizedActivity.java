package ro.mta.landmarkrecognitionapp;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.pm.ResolveInfo;
import android.net.Uri;
import android.os.Bundle;
import android.widget.ImageButton;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;
import androidx.viewpager2.widget.ViewPager2;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class ViewPhotoUnrecognizedActivity extends AppCompatActivity {
    private String unrecognizedImagesDirLocation;
    private UnrecognizedViewPager unrecognizedViewPager;
    private ViewPager2 viewPager;
    ArrayList<String> unrecognizedImagesList;
    private ImageButton deleteButton;
    private ImageButton shareImage;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_view_photo_unrecognized);
        unrecognizedImagesDirLocation = getFilesDir().getPath() + "/Unrecognized Images";
        viewPager = findViewById(R.id.photo_view_pager);
        File unrecognizedImagesDir = new File(unrecognizedImagesDirLocation);
        File[] files = unrecognizedImagesDir.listFiles();
        unrecognizedImagesList = new ArrayList<>();
        for (File file : files) {
            unrecognizedImagesList.add(file.getAbsolutePath());
        }
        if (unrecognizedImagesList.size() > 1)
            unrecognizedImagesList.sort(String::compareTo);
        unrecognizedViewPager = new UnrecognizedViewPager(unrecognizedImagesList, this);
        viewPager.setAdapter(unrecognizedViewPager);
        viewPager.setCurrentItem(unrecognizedViewPager.getItemCount() - 1);

        deleteButton = findViewById(R.id.delete_button);
        deleteButton.setOnClickListener(view -> {
            AlertDialog.Builder builder = new AlertDialog.Builder(this);
            builder.setCancelable(true);
            builder.setTitle("Delete Image");
            builder.setMessage("Do you want to delete this image?");
            builder.setPositiveButton(android.R.string.yes,
                    (dialog, which) -> {
                        int pos = viewPager.getCurrentItem();
                        if(pos==0)
                        {
                            if(unrecognizedImagesList.size() == 1)
                            {
                                deleteFile(pos);
                                finish();
                            }
                            else {
                                viewPager.setCurrentItem(pos + 1);
                                deleteFile(pos);
                            }
                        }
                        else {
                            viewPager.setCurrentItem(pos - 1);
                            deleteFile(pos);
                        }
                    });
            builder.setNegativeButton(android.R.string.no, (dialog, which) -> {
            }).setIcon(android.R.drawable.ic_dialog_alert);
            AlertDialog dialog = builder.create();
            dialog.show();
        });

        shareImage = findViewById(R.id.share_button);
        shareImage.setOnClickListener(view -> {
            Uri uriToImage = FileProvider.getUriForFile(this, "ro.mta.landmarkrecognitionapp.provider", new File(unrecognizedImagesList.get(viewPager.getCurrentItem())));
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
    }

    private void deleteFile(int pos){
        new File(unrecognizedImagesList.get(pos)).getAbsoluteFile().delete();
        unrecognizedImagesList.remove(pos);
        unrecognizedViewPager.notifyItemRemoved(pos);
    }
}