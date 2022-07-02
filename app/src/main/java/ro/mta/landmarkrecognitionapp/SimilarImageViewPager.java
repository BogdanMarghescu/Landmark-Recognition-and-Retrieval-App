package ro.mta.landmarkrecognitionapp;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;

public class SimilarImageViewPager extends RecyclerView.Adapter<SimilarImageViewPager.ViewHolder> {
    private String[] similarImagesURLs;
    private Context context;

    public SimilarImageViewPager(String[] similarImagesURLs, Context context) {
        this.similarImagesURLs = similarImagesURLs;
        this.context = context;
    }

    public class ViewHolder extends RecyclerView.ViewHolder {
        ImageView imageView;

        public ViewHolder(@NonNull View itemView) {
            super(itemView);
            imageView = itemView.findViewById(R.id.imageViewMain);
        }
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.image, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        Glide.with(context).load(similarImagesURLs[position]).into(holder.imageView);
    }

    @Override
    public int getItemCount() {
        return similarImagesURLs.length;
    }
}
