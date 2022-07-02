package ro.mta.landmarkrecognitionapp;

import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.content.pm.ResolveInfo;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageButton;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;
import androidx.viewpager2.widget.ViewPager2;

import com.google.android.material.button.MaterialButton;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.util.List;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class ViewPhotoUnrecognizedActivity extends AppCompatActivity {
    private List<UnrecognizedImages> unrecognizedImagesList;
    private LandmarkRecognitionDatabase landmarkRecognitionDatabase;
    private UnrecognizedImageViewPager unrecognizedImageViewPager;
    private ViewPager2 viewPager;
    private ImageButton deleteButton;
    private ImageButton shareImage;
    private MaterialButton recognizeImageButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_view_photo_unrecognized);
        landmarkRecognitionDatabase = LandmarkRecognitionDatabase.getInstance(this);
        viewPager = findViewById(R.id.photo_view_pager);
        unrecognizedImagesList = landmarkRecognitionDatabase.unrecognizedImagesDao().getUnrecognizedImagesList();
        unrecognizedImageViewPager = new UnrecognizedImageViewPager(unrecognizedImagesList, this);
        viewPager.setAdapter(unrecognizedImageViewPager);

        Intent intent = getIntent();
        viewPager.setCurrentItem(intent.getIntExtra("position", unrecognizedImageViewPager.getItemCount() - 1));

        deleteButton = findViewById(R.id.delete_button);
        deleteButton.setOnClickListener(view -> {
            AlertDialog.Builder builder = new AlertDialog.Builder(this);
            builder.setCancelable(true);
            builder.setTitle("Delete Image");
            builder.setMessage("Do you want to delete this image?");
            builder.setPositiveButton(android.R.string.yes,
                    (dialog, which) -> {
                        int pos = viewPager.getCurrentItem();
                        if (pos == 0) {
                            if (unrecognizedImagesList.size() == 1) {
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

        shareImage = findViewById(R.id.share_button);
        shareImage.setOnClickListener(view -> {
            Uri uriToImage = FileProvider.getUriForFile(this, "ro.mta.landmarkrecognitionapp.provider", new File(unrecognizedImagesList.get(viewPager.getCurrentItem()).getPath()));
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

        recognizeImageButton = findViewById(R.id.recognize_image_button);
        recognizeImageButton.setOnClickListener(view -> {
            int pos = viewPager.getCurrentItem();
            File imageToUpload = new File(unrecognizedImagesList.get(pos).getPath()).getAbsoluteFile();
            Uri uriToImage = FileProvider.getUriForFile(this, "ro.mta.landmarkrecognitionapp.provider", new File(unrecognizedImagesList.get(viewPager.getCurrentItem()).getPath()));
            Uploader uploader = NetworkClient.getRetrofit(this).create(Uploader.class);
            RequestBody requestFile = RequestBody.create(MediaType.parse(getContentResolver().getType(uriToImage)), imageToUpload);
            MultipartBody.Part body = MultipartBody.Part.createFormData("picture", imageToUpload.getName(), requestFile);
            Call<ResponseBody> call = uploader.upload(body);
            Toast.makeText(this, "Image " + imageToUpload.getName() + " sent to server for detection", Toast.LENGTH_SHORT).show();
            call.enqueue(new Callback<ResponseBody>() {
                @Override
                public void onResponse(@NonNull Call<ResponseBody> call, @NonNull Response<ResponseBody> response) {
                    Log.v("Upload", "success");
                    if (response.isSuccessful()) {
                        try {
                            assert response.body() != null;
                            JSONObject resp = new JSONObject(response.body().string());
                            String imageClassID = String.valueOf(resp.get("image_class_id"));
                            String landmarkName = String.valueOf(resp.get("landmark_name"));
                            String wikiURL = String.valueOf(resp.get("wiki_url"));
                            String similarImagesList = String.valueOf(resp.get("similar_images"));
                            String country = null;
                            String locality = null;
                            JSONArray address_components = (JSONArray) resp.get("address_components");
                            for (int i = 0; i < address_components.length(); i++) {
                                JSONObject address_component = (JSONObject) address_components.get(i);
                                if (String.valueOf(((JSONArray) address_component.get("types")).get(0)).equals("locality")) {
                                    locality = String.valueOf(address_component.get("long_name"));
                                } else if (String.valueOf(((JSONArray) address_component.get("types")).get(0)).equals("country")) {
                                    country = String.valueOf(address_component.get("long_name"));
                                }
                            }
                            double latitude = ((JSONObject) ((JSONObject) (resp.get("geometry"))).get("location")).getDouble("lat");
                            double longitude = ((JSONObject) ((JSONObject) (resp.get("geometry"))).get("location")).getDouble("lng");
                            Log.v("Upload", imageClassID);
                            Toast.makeText(ViewPhotoUnrecognizedActivity.this, "Image " + imageToUpload.getName() +
                                    "\nLandmark ID: " + imageClassID +
                                    "\nLandmark Name: " + landmarkName +
                                    "\nLandmark Country: " + country +
                                    "\nLandmark Locality: " + locality +
                                    "\nLandmark Latitude: " + latitude +
                                    "\nLandmark Longitude: " + longitude, Toast.LENGTH_SHORT).show();
                            String imageToUploadName = imageToUpload.getName();
                            imageToUploadName = imageToUploadName.substring(0, imageToUploadName.lastIndexOf("."));
                            File fileToUpload = new File(getFilesDir().getPath() + "/Recognized Images", imageToUpload.getName());
                            imageToUpload.renameTo(fileToUpload);
                            RecognizedImages recognizedImage = new RecognizedImages(fileToUpload.getAbsolutePath(), landmarkName, imageToUploadName, country, locality, wikiURL, similarImagesList, latitude, longitude);
                            if (landmarkRecognitionDatabase.recognizedImagesDao().getCountByPath(imageToUpload.getAbsolutePath()) == 0){
                                landmarkRecognitionDatabase.recognizedImagesDao().insertRecognizedImages(recognizedImage);
                                SharedPreferences sharedPreferences;
                                sharedPreferences = getSharedPreferences("sharedPref", MODE_PRIVATE);
                                SharedPreferences.Editor editor = sharedPreferences.edit();
                                editor.putBoolean(recognizedImage.getPath(), false);
                                editor.apply();
                            }
                            landmarkRecognitionDatabase.unrecognizedImagesDao().deleteUnrecognizedImages(unrecognizedImagesList.get(pos));
                            if (pos == 0) {
                                if (unrecognizedImagesList.size() == 1) {
                                    unrecognizedImagesList.remove(pos);
                                    unrecognizedImageViewPager.notifyItemRemoved(pos);
                                    finish();
                                } else {
                                    viewPager.setCurrentItem(pos + 1);
                                    unrecognizedImagesList.remove(pos);
                                    unrecognizedImageViewPager.notifyItemRemoved(pos);
                                }
                            } else {
                                viewPager.setCurrentItem(pos - 1);
                                unrecognizedImagesList.remove(pos);
                                unrecognizedImageViewPager.notifyItemRemoved(pos);
                            }
                        } catch (IOException | JSONException e) {
                            e.printStackTrace();
                            Log.e("Upload error:", e.getMessage());
                            Toast.makeText(ViewPhotoUnrecognizedActivity.this, e.getMessage(), Toast.LENGTH_SHORT).show();
                        }
                    }
                }

                @Override
                public void onFailure(@NonNull Call<ResponseBody> call, @NonNull Throwable t) {
                    Log.e("Upload error:", t.getMessage());
                    Toast.makeText(ViewPhotoUnrecognizedActivity.this, t.getMessage(), Toast.LENGTH_SHORT).show();
                }
            });
        });
    }

    private void deleteFile(int pos) {
        landmarkRecognitionDatabase.unrecognizedImagesDao().deleteUnrecognizedImages(unrecognizedImagesList.get(pos));
        new File(unrecognizedImagesList.get(pos).getPath()).getAbsoluteFile().delete();
        unrecognizedImagesList.remove(pos);
        unrecognizedImageViewPager.notifyItemRemoved(pos);
    }
}