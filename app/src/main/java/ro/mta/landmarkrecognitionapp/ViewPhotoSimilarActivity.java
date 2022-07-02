package ro.mta.landmarkrecognitionapp;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.os.Bundle;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.viewpager2.widget.ViewPager2;

public class ViewPhotoSimilarActivity extends AppCompatActivity {
    private ViewPager2 viewPager;
    private String[] similarImagesURLs;
    private SimilarImageViewPager similarImageViewPager;
    private TextView similarImageCountTextView;

    @SuppressLint("SetTextI18n")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_view_photo_similar);
        viewPager = findViewById(R.id.photo_view_pager_similar);
        Intent intent = getIntent();
        if (intent.hasExtra("similar_images")) {
            String similarImagesList = intent.getStringExtra("similar_images");
            similarImagesURLs = similarImagesList.split(" ");
            similarImageViewPager = new SimilarImageViewPager(similarImagesURLs, this);
            viewPager.setAdapter(similarImageViewPager);
            viewPager.setCurrentItem(0);
            similarImageCountTextView = findViewById(R.id.similar_image_count);
            similarImageCountTextView.setText("Image " + (viewPager.getCurrentItem() + 1) + " out of " + similarImagesURLs.length);
            viewPager.registerOnPageChangeCallback(new ViewPager2.OnPageChangeCallback() {
                @Override
                public void onPageSelected(int position) {
                    super.onPageSelected(position);
                    similarImageCountTextView.setText("Image " + (viewPager.getCurrentItem() + 1) + " out of " + similarImagesURLs.length);
                }
            });
        }
    }
}