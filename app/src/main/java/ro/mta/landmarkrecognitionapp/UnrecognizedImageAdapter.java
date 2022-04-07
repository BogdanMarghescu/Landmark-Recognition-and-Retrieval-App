package ro.mta.landmarkrecognitionapp;

import android.content.Context;
import android.content.Intent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.GridView;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.bumptech.glide.Glide;

import java.util.ArrayList;

public class UnrecognizedImageAdapter extends ArrayAdapter<String> {
    private Context context;
    private ArrayList<String> imagePaths;

    public UnrecognizedImageAdapter(@NonNull Context context, @NonNull ArrayList<String> imagePaths) {
        super(context, R.layout.image, imagePaths);
        this.context = context;
        this.imagePaths = imagePaths;
    }

    @Override
    public int getCount() {
        return imagePaths.size();
    }

    @Nullable
    @Override
    public String getItem(int position) {
        return imagePaths.get(position);
    }

    @Override
    public long getItemId(int position) {
        return 0;
    }

    @NonNull
    @Override
    public View getView(int position, @Nullable View convertView, @NonNull ViewGroup parent) {
        ImageView imageView;
        if (convertView == null) {
            imageView = new ImageView(this.context);
            imageView.setLayoutParams(new GridView.LayoutParams(170, 170));
            imageView.setScaleType(ImageView.ScaleType.CENTER_CROP);
        } else {
            imageView = (ImageView) convertView;
        }
        imageView.setClickable(true);
        imageView.setOnClickListener(view -> {
            Intent intent = new Intent(context, ViewPhotoUnrecognizedActivity.class);
            intent.putExtra("position", position);
            context.startActivity(intent);
        });
        String filename = getItem(position);
        Glide.with(context).load(filename).into(imageView);
        return imageView;
    }
}
