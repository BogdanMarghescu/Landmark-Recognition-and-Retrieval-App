package ro.mta.landmarkrecognitionapp.ui.main;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.fragment.app.Fragment;

import ro.mta.landmarkrecognitionapp.R;

public class RecognizedImagesFragment extends Fragment {
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
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_recognized_images, container, false);
    }
}