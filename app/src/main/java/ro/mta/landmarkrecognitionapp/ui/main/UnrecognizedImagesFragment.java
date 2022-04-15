package ro.mta.landmarkrecognitionapp.ui.main;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.GridView;

import androidx.fragment.app.Fragment;

import java.util.List;

import ro.mta.landmarkrecognitionapp.GalleryUnrecognizedImageAdapter;
import ro.mta.landmarkrecognitionapp.LandmarkRecognitionDatabase;
import ro.mta.landmarkrecognitionapp.R;
import ro.mta.landmarkrecognitionapp.UnrecognizedImages;

public class UnrecognizedImagesFragment extends Fragment {
    private GridView imageGrid;
    private List<UnrecognizedImages> unrecognizedImagesList;
    private LandmarkRecognitionDatabase landmarkRecognitionDatabase;
    private GalleryUnrecognizedImageAdapter galleryUnrecognizedImageAdapter;

    public UnrecognizedImagesFragment() {
    }

    public static UnrecognizedImagesFragment newInstance() {
        UnrecognizedImagesFragment fragment = new UnrecognizedImagesFragment();
        return fragment;
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_unrecognized_images, container, false);
        landmarkRecognitionDatabase = LandmarkRecognitionDatabase.getInstance(view.getContext());
        imageGrid = view.findViewById(R.id.unrecognized_gallery);
        unrecognizedImagesList = landmarkRecognitionDatabase.unrecognizedImagesDao().getUnrecognizedImagesList();
        galleryUnrecognizedImageAdapter = new GalleryUnrecognizedImageAdapter(requireContext(), unrecognizedImagesList);
        imageGrid.setAdapter(galleryUnrecognizedImageAdapter);
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