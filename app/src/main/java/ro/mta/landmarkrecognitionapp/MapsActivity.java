package ro.mta.landmarkrecognitionapp;

import android.content.Intent;
import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.fragment.app.FragmentActivity;

import com.google.android.gms.maps.CameraUpdateFactory;
import com.google.android.gms.maps.GoogleMap;
import com.google.android.gms.maps.OnMapReadyCallback;
import com.google.android.gms.maps.SupportMapFragment;
import com.google.android.gms.maps.model.LatLng;
import com.google.android.gms.maps.model.Marker;
import com.google.android.gms.maps.model.MarkerOptions;

import java.util.ArrayList;
import java.util.List;

import ro.mta.landmarkrecognitionapp.databinding.ActivityMapsBinding;

public class MapsActivity extends FragmentActivity implements OnMapReadyCallback {
    private GoogleMap map;
    private Marker marker;
    private ActivityMapsBinding binding;
    LandmarkRecognitionDatabase landmarkRecognitionDatabase;
    List<RecognizedImages> recognizedImagesList;
    ArrayList<String> recognizedImagesListString;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMapsBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
        SupportMapFragment mapFragment = (SupportMapFragment) getSupportFragmentManager().findFragmentById(R.id.map);
        assert mapFragment != null;
        mapFragment.getMapAsync(this);
        landmarkRecognitionDatabase = LandmarkRecognitionDatabase.getInstance(this);
        recognizedImagesList = landmarkRecognitionDatabase.recognizedImagesDao().getRecognizedImagesList();
    }

    @Override
    public void onMapReady(@NonNull GoogleMap googleMap) {
        map = googleMap;
        LatLng coords;
        for (RecognizedImages image : recognizedImagesList) {
            coords = new LatLng(image.getLatitude(), image.getLongitude());
            marker = map.addMarker(new MarkerOptions().position(coords).title(image.getLandmarkName()));
            marker.setTag(recognizedImagesList.indexOf(image));
        }
        Intent intent = getIntent();
        if (intent.hasExtra("landmark")) {
            RecognizedImages landmark = (RecognizedImages) intent.getParcelableExtra("landmark");
            coords = new LatLng(landmark.getLatitude(), landmark.getLongitude());
            map.animateCamera(CameraUpdateFactory.newLatLngZoom(coords, 14));
        } else {
            coords = new LatLng(recognizedImagesList.get(0).getLatitude(), recognizedImagesList.get(0).getLongitude());
            map.moveCamera(CameraUpdateFactory.newLatLng(coords));
        }
        map.setOnMarkerClickListener(new GoogleMap.OnMarkerClickListener() {
            @Override
            public boolean onMarkerClick(@NonNull Marker marker) {
                int position = (int) marker.getTag();
                String landmarkName = recognizedImagesList.get(position).getLandmarkName();
                List<RecognizedImages> recognizedImagesListByLandmark = landmarkRecognitionDatabase.recognizedImagesDao().getImagesByLandmark(landmarkName);
                recognizedImagesListString = new ArrayList<>();
                for (RecognizedImages recognizedImage : recognizedImagesListByLandmark)
                    recognizedImagesListString.add(recognizedImage.getPath());
                Intent intentImages = new Intent(getApplicationContext(), ViewPhotoRecognizedActivity.class);
                intentImages.putExtra("image_list", recognizedImagesListString);
                startActivity(intentImages);
                return false;
            }
        });
    }
}