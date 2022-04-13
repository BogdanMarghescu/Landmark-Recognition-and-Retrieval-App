package ro.mta.landmarkrecognitionapp;

import android.annotation.SuppressLint;
import android.content.DialogInterface;
import android.content.Intent;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.widget.Toast;

import androidx.annotation.StringRes;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.viewpager2.widget.ViewPager2;

import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.android.material.tabs.TabLayout;
import com.google.android.material.tabs.TabLayoutMediator;

import java.io.File;

import ro.mta.landmarkrecognitionapp.ui.main.SectionsPagerAdapter;

public class MainActivity extends AppCompatActivity {
    @StringRes
    private static final int[] TAB_TITLES = new int[]{R.string.tab_text_1, R.string.tab_text_2, R.string.tab_text_3};
    ViewPager2 viewPager;
    TabLayout tabs;
    SectionsPagerAdapter sectionsPagerAdapter;
    private String unrecognizedImagesDirLocation;
    private String recognizedImagesDirLocation;
    boolean pauseFlag = false;

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
                Intent intent = new Intent(getApplicationContext(), MapsActivity.class);
                startActivity(intent);
                return true;
            default:
                return super.onOptionsItemSelected(item);
        }
    }

    private void showAlertDialog() {
        AlertDialog.Builder alertDialog = new AlertDialog.Builder(MainActivity.this);
        alertDialog.setTitle("Sort Images by:");
        String[] items = {"Date", "Country", "Locality", "Landmark", "Favorites"};
        int checkedItem = 0;
        alertDialog.setSingleChoiceItems(items, checkedItem, new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                switch (which) {
                    case 0:
                        Toast.makeText(MainActivity.this, "Clicked on Date", Toast.LENGTH_SHORT).show();
                        break;
                    case 1:
                        Toast.makeText(MainActivity.this, "Clicked on Country", Toast.LENGTH_SHORT).show();
                        break;
                    case 2:
                        Toast.makeText(MainActivity.this, "Clicked on Locality", Toast.LENGTH_SHORT).show();
                        break;
                    case 3:
                        Toast.makeText(MainActivity.this, "Clicked on Landmark", Toast.LENGTH_SHORT).show();
                        break;
                    case 4:
                        Toast.makeText(MainActivity.this, "Clicked on Favorites", Toast.LENGTH_SHORT).show();
                        break;
                }
            }
        });
        AlertDialog alert = alertDialog.create();
        alert.setCanceledOnTouchOutside(false);
        alert.show();
    }
}