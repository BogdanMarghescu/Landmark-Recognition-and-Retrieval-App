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

import java.util.List;

public class GalleryRecognizedImageAdapter extends ArrayAdapter<RecognizedImages> {
    private Context context;
    private List<RecognizedImages> recognizedImagesList;

    public GalleryRecognizedImageAdapter(@NonNull Context context, @NonNull List<RecognizedImages> imagePaths) {
        super(context, R.layout.image, imagePaths);
        this.context = context;
        this.recognizedImagesList = imagePaths;
    }

    @Override
    public int getCount() {
        return recognizedImagesList.size();
    }

    @Nullable
    @Override
    public RecognizedImages getItem(int position) {
        return recognizedImagesList.get(position);
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
            Intent intent = new Intent(context, ViewPhotoRecognizedActivity.class);
            intent.putExtra("position", position);
            context.startActivity(intent);
        });
        RecognizedImages recognizedImage = getItem(position);
        assert recognizedImage != null;
        Glide.with(context).load(recognizedImage.getPath()).into(imageView);
        return imageView;
    }
}
