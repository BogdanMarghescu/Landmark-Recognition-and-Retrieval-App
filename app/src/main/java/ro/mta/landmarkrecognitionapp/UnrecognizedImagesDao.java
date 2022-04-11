package ro.mta.landmarkrecognitionapp;

import androidx.room.Dao;
import androidx.room.Delete;
import androidx.room.Insert;
import androidx.room.OnConflictStrategy;
import androidx.room.Query;
import androidx.room.Update;

import java.util.List;

@Dao
public interface UnrecognizedImagesDao {
    @Query("SELECT * FROM unrecognized_images")
    List<UnrecognizedImages> getRecognizedImagesList();

    @Query("SELECT COUNT(*) FROM unrecognized_images WHERE path = :myPath")
    int getCountByPath(String myPath);

    @Query("SELECT * FROM unrecognized_images WHERE path = :myPath")
    UnrecognizedImages getImageByPath(String myPath);

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    void insertUnrecognizedImages(UnrecognizedImages unrecognizedImages);

    @Update
    void updateUnrecognizedImages(UnrecognizedImages unrecognizedImages);

    @Delete
    void deleteUnrecognizedImages(UnrecognizedImages unrecognizedImages);
}
