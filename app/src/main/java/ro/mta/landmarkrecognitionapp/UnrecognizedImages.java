package ro.mta.landmarkrecognitionapp;

import androidx.room.ColumnInfo;
import androidx.room.Entity;
import androidx.room.Ignore;
import androidx.room.PrimaryKey;

@Entity(tableName = "unrecognized_images")
public class UnrecognizedImages {
    @PrimaryKey(autoGenerate = true)
    private int id;
    @ColumnInfo(name = "path")
    private String path;

    public UnrecognizedImages(int id, String path) {
        this.id = id;
        this.path = path;
    }

    @Ignore
    public UnrecognizedImages(String path) {
        this.path = path;
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
}
