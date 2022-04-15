package ro.mta.landmarkrecognitionapp.ui.main;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.GridView;
import android.widget.TextView;

import androidx.fragment.app.Fragment;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import ro.mta.landmarkrecognitionapp.GalleryRecognizedImageAdapter;
import ro.mta.landmarkrecognitionapp.LandmarkRecognitionDatabase;
import ro.mta.landmarkrecognitionapp.R;
import ro.mta.landmarkrecognitionapp.RecognizedImages;

public class RecognizedImagesFragment extends Fragment {
    private GridView imageGrid;
    private TextView sortTypeTextView;
    private LandmarkRecognitionDatabase landmarkRecognitionDatabase;
    private GalleryRecognizedImageAdapter galleryRecognizedImageAdapter;
    private List<RecognizedImages> recognizedImagesList;
    private String sortType;

    public RecognizedImagesFragment() {
    }

    public static RecognizedImagesFragment newInstance() {
        RecognizedImagesFragment fragment = new RecognizedImagesFragment();
        return fragment;
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
    }

    @SuppressLint("SetTextI18n")
    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_recognized_images, container, false);
        landmarkRecognitionDatabase = LandmarkRecognitionDatabase.getInstance(view.getContext());
        imageGrid = view.findViewById(R.id.recognized_gallery);
        SharedPreferences sharedPreferences;
        sharedPreferences = view.getContext().getSharedPreferences("sharedPref", Context.MODE_PRIVATE);
        sortType = sharedPreferences.getString("Sort Type", "Date");
        sortTypeTextView = view.findViewById(R.id.sort_type);
        sortTypeTextView.setText("Images sorted by: " + sortType);
        switch (sortType) {
            case "Date":
                recognizedImagesList = landmarkRecognitionDatabase.recognizedImagesDao().getRecognizedImagesListOrderDate();
                break;
            case "Country":
                recognizedImagesList = landmarkRecognitionDatabase.recognizedImagesDao().getRecognizedImagesListOrderCountry();
                break;
            case "Locality":
                recognizedImagesList = landmarkRecognitionDatabase.recognizedImagesDao().getRecognizedImagesListOrderLocality();
                break;
            case "Landmark":
                recognizedImagesList = landmarkRecognitionDatabase.recognizedImagesDao().getRecognizedImagesListOrderLandmark();
                break;
            case "Favorites":
                recognizedImagesList = new ArrayList<>();
                Map<String, ?> allEntries = sharedPreferences.getAll();
                for (Map.Entry<String, ?> entry : allEntries.entrySet()) {
                    if (entry.getValue().getClass().equals(Boolean.class) && (boolean) entry.getValue()) {
                        RecognizedImages favImage = landmarkRecognitionDatabase.recognizedImagesDao().getImageByPath(entry.getKey());
                        if (favImage != null) recognizedImagesList.add(favImage);
                    }
                }
                break;
        }
        galleryRecognizedImageAdapter = new GalleryRecognizedImageAdapter(requireContext(), recognizedImagesList);
        imageGrid.setAdapter(galleryRecognizedImageAdapter);
        return view;
    }


    @Override
    public void setUserVisibleHint(boolean isVisibleToUser) {
        super.setUserVisibleHint(isVisibleToUser);
        if (isVisibleToUser) {
            assert getFragmentManager() != null;
            getFragmentManager().beginTransaction().detach(this).attach(this).commit();
        }
    }
}