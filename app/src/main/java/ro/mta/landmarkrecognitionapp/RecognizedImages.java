package ro.mta.landmarkrecognitionapp;


import androidx.room.ColumnInfo;
import androidx.room.Entity;
import androidx.room.Ignore;
import androidx.room.PrimaryKey;

@Entity(tableName = "recognized_images")
public class RecognizedImages {
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
    @ColumnInfo(name = "latitude")
    private double latitude;
    @ColumnInfo(name = "longitude")
    private double longitude;

    public RecognizedImages(int id, String path, String landmarkName, String dateTaken, String country, String locality, double latitude, double longitude) {
        this.id = id;
        this.path = path;
        this.landmarkName = landmarkName;
        this.dateTaken = dateTaken;
        this.country = country;
        this.locality = locality;
        this.latitude = latitude;
        this.longitude = longitude;
    }

    @Ignore
    public RecognizedImages(String path, String landmarkName, String dateTaken, String country, String locality, double latitude, double longitude) {
        this.path = path;
        this.landmarkName = landmarkName;
        this.dateTaken = dateTaken;
        this.country = country;
        this.locality = locality;
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
}
