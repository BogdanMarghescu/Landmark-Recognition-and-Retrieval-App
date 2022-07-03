package ro.mta.landmarkrecognitionapp.ui.main;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.fragment.app.Fragment;

import com.github.mikephil.charting.charts.PieChart;
import com.github.mikephil.charting.components.Legend;
import com.github.mikephil.charting.data.PieData;
import com.github.mikephil.charting.data.PieDataSet;
import com.github.mikephil.charting.data.PieEntry;
import com.github.mikephil.charting.utils.ColorTemplate;

import java.util.ArrayList;
import java.util.Map;

import ro.mta.landmarkrecognitionapp.CountryCount;
import ro.mta.landmarkrecognitionapp.LandmarkCount;
import ro.mta.landmarkrecognitionapp.LandmarkRecognitionDatabase;
import ro.mta.landmarkrecognitionapp.LocalityCount;
import ro.mta.landmarkrecognitionapp.R;

public class StatisticsFragment extends Fragment {
    LandmarkRecognitionDatabase landmarkRecognitionDatabase;
    PieChart pieChart_image_count_recognition;
    PieChart pieChart_favorites_count;
    PieChart pieChart_countries_count;
    PieChart pieChart_locality_count;
    PieChart pieChart_landmark_count;

    public StatisticsFragment() {
    }

    public static StatisticsFragment newInstance() {
        StatisticsFragment fragment = new StatisticsFragment();
        return fragment;
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_statistics, container, false);
        landmarkRecognitionDatabase = LandmarkRecognitionDatabase.getInstance(view.getContext());
        ArrayList<PieEntry> pieEntries;
        pieChart_image_count_recognition = view.findViewById(R.id.piechart_image_count_recognition);
        int recognizedImagesCount = landmarkRecognitionDatabase.recognizedImagesDao().getCount();
        int unrecognizedImagesCount = landmarkRecognitionDatabase.unrecognizedImagesDao().getCount();
        pieEntries = new ArrayList<>();
        pieEntries.add(new PieEntry(recognizedImagesCount, "Recognized"));
        pieEntries.add(new PieEntry(unrecognizedImagesCount, "Unrecognized"));
        setPieChart(pieChart_image_count_recognition, pieEntries);

        pieChart_favorites_count = view.findViewById(R.id.piechart_favorite_count);
        SharedPreferences sharedPreferences = view.getContext().getSharedPreferences("sharedPref", Context.MODE_PRIVATE);
        Map<String, ?> allEntries = sharedPreferences.getAll();
        int numFavorites = 0, numNotFavorites = 0;
        for (Map.Entry<String, ?> entry : allEntries.entrySet()) {
            if (entry.getValue().getClass().equals(Boolean.class)) {
                if ((boolean) entry.getValue())
                    numFavorites++;
                else
                    numNotFavorites++;
            }
        }
        pieEntries = new ArrayList<>();
        pieEntries.add(new PieEntry(numFavorites, "Favorite"));
        pieEntries.add(new PieEntry(numNotFavorites, "Not Favorite"));
        setPieChart(pieChart_favorites_count, pieEntries);

        ArrayList<CountryCount> countryCountArrayList = (ArrayList<CountryCount>) landmarkRecognitionDatabase.recognizedImagesDao().getCountries();
        pieChart_countries_count = view.findViewById(R.id.piechart_country);
        pieEntries = new ArrayList<>();
        for (CountryCount countryCount : countryCountArrayList)
            pieEntries.add(new PieEntry(countryCount.count, countryCount.country));
        setPieChart(pieChart_countries_count, pieEntries);

        ArrayList<LocalityCount> localityCountArrayList = (ArrayList<LocalityCount>) landmarkRecognitionDatabase.recognizedImagesDao().getLocalities();
        pieChart_locality_count = view.findViewById(R.id.piechart_locality);
        pieEntries = new ArrayList<>();
        for (LocalityCount localityCount : localityCountArrayList)
            pieEntries.add(new PieEntry(localityCount.count, localityCount.locality));
        setPieChart(pieChart_locality_count, pieEntries);

        ArrayList<LandmarkCount> landmarkCountArrayList = (ArrayList<LandmarkCount>) landmarkRecognitionDatabase.recognizedImagesDao().getLandmarks();
        pieChart_landmark_count = view.findViewById(R.id.piechart_landmark);
        pieEntries = new ArrayList<>();
        for (LandmarkCount landmarkCount : landmarkCountArrayList)
            pieEntries.add(new PieEntry(landmarkCount.count, landmarkCount.landmark_name));
        setPieChart(pieChart_landmark_count, pieEntries);

        return view;
    }

    private void setPieChart(PieChart pieChart, ArrayList<PieEntry> pieEntries) {
        PieDataSet pieDataSet = new PieDataSet(pieEntries, "");
        pieDataSet.setColors(ColorTemplate.MATERIAL_COLORS);
        pieDataSet.setXValuePosition(PieDataSet.ValuePosition.INSIDE_SLICE);
        pieDataSet.setYValuePosition(PieDataSet.ValuePosition.INSIDE_SLICE);
        pieDataSet.setValueTextSize(16);
        pieDataSet.setSliceSpace(3);
        PieData pieData = new PieData(pieDataSet);
        pieData.setValueFormatter((value, entry, dataSetIndex, viewPortHandler) -> String.valueOf((int) value));
        pieChart.setData(pieData);
        pieChart.setDescription(null);
        pieChart.setEntryLabelColor(getResources().getColor(R.color.black));
        Legend legend = pieChart.getLegend();
        legend.setTextSize(13);
        legend.setDrawInside(false);
        legend.setTextColor(getResources().getColor(R.color.white));
        legend.setWordWrapEnabled(true);
        pieChart.animateXY(2000, 2000);
        pieChart.invalidate();
    }
}