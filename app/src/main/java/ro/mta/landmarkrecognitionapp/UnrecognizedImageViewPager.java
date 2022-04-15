package ro.mta.landmarkrecognitionapp;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;

import java.util.List;

public class UnrecognizedImageViewPager extends RecyclerView.Adapter<UnrecognizedImageViewPager.ViewHolder> {
    private List<UnrecognizedImages> imageList;
    private Context context;

    public UnrecognizedImageViewPager(List<UnrecognizedImages> imageList, Context context) {
        this.imageList = imageList;
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
        Glide.with(context).load(imageList.get(position).getPath()).into(holder.imageView);
    }

    @Override
    public int getItemCount() {
        return imageList.size();
    }
}