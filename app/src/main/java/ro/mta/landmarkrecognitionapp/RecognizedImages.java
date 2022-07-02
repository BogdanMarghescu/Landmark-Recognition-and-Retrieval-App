package ro.mta.landmarkrecognitionapp;


import android.os.Parcel;
import android.os.Parcelable;

import androidx.room.ColumnInfo;
import androidx.room.Entity;
import androidx.room.Ignore;
import androidx.room.PrimaryKey;

@Entity(tableName = "recognized_images")
public class RecognizedImages implements Parcelable {
    @PrimaryKey(autoGenerate = true)
    private int id;
    @ColumnInfo(name = "path")
    private String path;
    @ColumnInfo(name = "landmark_name")
    private String landmarkName;
    @ColumnInfo(name = "date_taken")
    private String dateTaken;
    @ColumnInfo(name = "country")
    private String country;
    @ColumnInfo(name = "locality")
    private String locality;
    @ColumnInfo(name = "wiki_url")
    private String wiki_url;
    @ColumnInfo(name = "similar_images")
    private String similar_images;
    @ColumnInfo(name = "latitude")
    private double latitude;
    @ColumnInfo(name = "longitude")
    private double longitude;

    public RecognizedImages(int id, String path, String landmarkName, String dateTaken, String country, String locality, String wiki_url, String similar_images, double latitude, double longitude) {
        this.id = id;
        this.path = path;
        this.landmarkName = landmarkName;
        this.dateTaken = dateTaken;
        this.country = country;
        this.locality = locality;
        this.wiki_url = wiki_url;
        this.similar_images = similar_images;
        this.latitude = latitude;
        this.longitude = longitude;
    }

    @Ignore
    public RecognizedImages(String path, String landmarkName, String dateTaken, String country, String locality, String wiki_url, String similar_images, double latitude, double longitude) {
        this.path = path;
        this.landmarkName = landmarkName;
        this.dateTaken = dateTaken;
        this.country = country;
        this.locality = locality;
        this.wiki_url = wiki_url;
        this.similar_images = similar_images;
        this.latitude = latitude;
        this.longitude = longitude;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public String getPath() {
        return path;
    }

    public void setPath(String path) {
        this.path = path;
    }

    public String getLandmarkName() {
        return landmarkName;
    }

    public void setLandmarkName(String landmarkName) {
        this.landmarkName = landmarkName;
    }

    public String getDateTaken() {
        return dateTaken;
    }

    public void setDateTaken(String dateTaken) {
        this.dateTaken = dateTaken;
    }

    public String getCountry() {
        return country;
    }

    public void setCountry(String country) {
        this.country = country;
    }

    public String getLocality() {
        return locality;
    }

    public void setLocality(String locality) {
        this.locality = locality;
    }

    public String getWiki_url() {
        return wiki_url;
    }

    public void setWiki_url(String wiki_url) {
        this.wiki_url = wiki_url;
    }

    public String getSimilar_images() {
        return similar_images;
    }

    public void setSimilar_images(String similar_images) {
        this.similar_images = similar_images;
    }

    public double getLatitude() {
        return latitude;
    }

    public void setLatitude(double latitude) {
        this.latitude = latitude;
    }

    public double getLongitude() {
        return longitude;
    }

    public void setLongitude(double longitude) {
        this.longitude = longitude;
    }

    @Override
    public int describeContents() {
        return 0;
    }

    @Override
    public void writeToParcel(Parcel dest, int flags) {
        dest.writeInt(this.id);
        dest.writeString(this.path);
        dest.writeString(this.landmarkName);
        dest.writeString(this.dateTaken);
        dest.writeString(this.country);
        dest.writeString(this.locality);
        dest.writeString(this.wiki_url);
        dest.writeString(this.similar_images);
        dest.writeDouble(this.latitude);
        dest.writeDouble(this.longitude);
    }

    public void readFromParcel(Parcel source) {
        this.id = source.readInt();
        this.path = source.readString();
        this.landmarkName = source.readString();
        this.dateTaken = source.readString();
        this.country = source.readString();
        this.locality = source.readString();
        this.wiki_url = source.readString();
        this.similar_images = source.readString();
        this.latitude = source.readDouble();
        this.longitude = source.readDouble();
    }

    protected RecognizedImages(Parcel in) {
        this.id = in.readInt();
        this.path = in.readString();
        this.landmarkName = in.readString();
        this.dateTaken = in.readString();
        this.country = in.readString();
        this.locality = in.readString();
        this.wiki_url = in.readString();
        this.similar_images = in.readString();
        this.latitude = in.readDouble();
        this.longitude = in.readDouble();
    }

    public static final Parcelable.Creator<RecognizedImages> CREATOR = new Parcelable.Creator<RecognizedImages>() {
        @Override
        public RecognizedImages createFromParcel(Parcel source) {
            return new RecognizedImages(source);
        }

        @Override
        public RecognizedImages[] newArray(int size) {
            return new RecognizedImages[size];
        }
    };
}
