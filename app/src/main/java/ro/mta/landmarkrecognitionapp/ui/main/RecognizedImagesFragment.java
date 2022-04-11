package ro.mta.landmarkrecognitionapp.ui.main;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.GridView;

import androidx.fragment.app.Fragment;

import java.io.File;
import java.util.ArrayList;

import ro.mta.landmarkrecognitionapp.GalleryRecognizedImageAdapter;
import ro.mta.landmarkrecognitionapp.R;

public class RecognizedImagesFragment extends Fragment {
    private GridView imageGrid;
    private ArrayList<String> recognizedImagesList;
    private String recognizedImagesDirLocation;
    private GalleryRecognizedImageAdapter galleryRecognizedImageAdapter;

    public RecognizedImagesFragment() {
        // Required empty public constructor
    }

    public static RecognizedImagesFragment newInstance(String param1, String param2) {
        RecognizedImagesFragment fragment = new RecognizedImagesFragment();
        return fragment;
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_recognized_images, container, false);
        imageGrid = view.findViewById(R.id.recognized_gallery);
        recognizedImagesDirLocation = requireContext().getFilesDir().getPath() + "/Recognized Images";
        File unrecognizedImagesDir = new File(recognizedImagesDirLocation);
        File[] files = unrecognizedImagesDir.listFiles();
        recognizedImagesList = new ArrayList<>();
        for (File file : files) {
            recognizedImagesList.add(file.getAbsolutePath());
        }
        if (recognizedImagesList.size() > 1)
            recognizedImagesList.sort(String::compareTo);
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