package ro.mta.landmarkrecognitionapp;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;

import androidx.annotation.StringRes;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.viewpager2.widget.ViewPager2;

import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.android.material.tabs.TabLayout;
import com.google.android.material.tabs.TabLayoutMediator;

import java.io.File;
import java.util.Arrays;
import java.util.List;

import ro.mta.landmarkrecognitionapp.ui.main.SectionsPagerAdapter;

public class MainActivity extends AppCompatActivity {
    @StringRes
    private static final int[] TAB_TITLES = new int[]{R.string.tab_text_1, R.string.tab_text_2, R.string.tab_text_3};
    ViewPager2 viewPager;
    TabLayout tabs;
    SectionsPagerAdapter sectionsPagerAdapter;
    private String unrecognizedImagesDirLocation;
    private String recognizedImagesDirLocation;
    private boolean pauseFlag = false;
    private SharedPreferences sharedPreferences;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        unrecognizedImagesDirLocation = getFilesDir().getPath() + "/Unrecognized Images";
        recognizedImagesDirLocation = getFilesDir().getPath() + "/Recognized Images";
        File unrecognizedImagesDir = new File(unrecognizedImagesDirLocation);
        File recognizedImagesDir = new File(recognizedImagesDirLocation);
        if (!unrecognizedImagesDir.exists())
            unrecognizedImagesDir.mkdirs();
        if (!recognizedImagesDir.exists())
            recognizedImagesDir.mkdirs();
        sharedPreferences = getSharedPreferences("sharedPref", MODE_PRIVATE);
        if (!sharedPreferences.contains("Sort Type")) {
            @SuppressLint("CommitPrefEdits") SharedPreferences.Editor editor = sharedPreferences.edit();
            editor.putString("Sort Type", "Date");
            editor.apply();
        }
        sectionsPagerAdapter = new SectionsPagerAdapter(this);
        viewPager = findViewById(R.id.view_pager_galleries);
        viewPager.setAdapter(sectionsPagerAdapter);
        tabs = findViewById(R.id.tabs);
        new TabLayoutMediator(tabs, viewPager, (tab, position) -> tab.setText(TAB_TITLES[position])).attach();
        FloatingActionButton fab = findViewById(R.id.fab_camera);
        fab.setOnClickListener(view -> {
            Intent intent = new Intent(getApplicationContext(), CameraActivity.class);
            startActivity(intent);
        });
        setSupportActionBar(findViewById(R.id.toolbar));
    }

    @Override
    protected void onStart() {
        super.onStart();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (pauseFlag) {
            recreate();
            pauseFlag = false;
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        pauseFlag = true;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.menu, menu);
        return true;
    }

    @SuppressLint("NonConstantResourceId")
    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.item1:
                showAlertDialog();
                return true;
            case R.id.item2:
                File recognizedImagesDir = new File(recognizedImagesDirLocation);
                if (recognizedImagesDir.length() > 0) {
                    Intent intent = new Intent(getApplicationContext(), MapsActivity.class);
                    startActivity(intent);
                } else {
                    AlertDialog.Builder builder = new AlertDialog.Builder(this);
                    builder.setCancelable(true);
                    builder.setTitle("Empty Map");
                    builder.setMessage("There are no landmarks to be localized, try visiting this menu when you have recognized images.");
                    builder.setPositiveButton(android.R.string.ok, null);
                    builder.setIcon(android.R.drawable.ic_dialog_alert);
                    AlertDialog dialog = builder.create();
                    dialog.show();
                }
                return true;
            default:
                return super.onOptionsItemSelected(item);
        }
    }

    private void showAlertDialog() {
        AlertDialog.Builder alertDialog = new AlertDialog.Builder(MainActivity.this);
        alertDialog.setTitle("Sort Images by:");
        String[] items = {"Date", "Country", "Locality", "Landmark", "Favorites"};
        List<Object> stringArrayList = Arrays.asList(items);
        int checkedItem = stringArrayList.indexOf(sharedPreferences.getString("Sort Type", "Date"));
        @SuppressLint("CommitPrefEdits") SharedPreferences.Editor editor = sharedPreferences.edit();
        alertDialog.setSingleChoiceItems(items, checkedItem, (dialog, which) -> {
            editor.putString("Sort Type", items[which]);
            editor.apply();
            recreate();
            viewPager.setCurrentItem(0);
        });
        AlertDialog alert = alertDialog.create();
        alert.setCanceledOnTouchOutside(false);
        alert.show();
    }
}