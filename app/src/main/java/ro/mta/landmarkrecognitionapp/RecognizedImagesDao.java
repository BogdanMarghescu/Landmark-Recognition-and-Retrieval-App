package ro.mta.landmarkrecognitionapp;

import androidx.room.Dao;
import androidx.room.Delete;
import androidx.room.Insert;
import androidx.room.OnConflictStrategy;
import androidx.room.Query;
import androidx.room.Update;

import java.util.List;

@Dao
public interface RecognizedImagesDao {
    @Query("SELECT * FROM recognized_images")
    List<RecognizedImages> getRecognizedImagesList();

    @Query("SELECT * FROM recognized_images ORDER BY date_taken")
    List<RecognizedImages> getRecognizedImagesListOrderDate();

    @Query("SELECT * FROM recognized_images ORDER BY country")
    List<RecognizedImages> getRecognizedImagesListOrderCountry();

    @Query("SELECT * FROM recognized_images ORDER BY locality")
    List<RecognizedImages> getRecognizedImagesListOrderLocality();

    @Query("SELECT * FROM recognized_images ORDER BY landmark_name")
    List<RecognizedImages> getRecognizedImagesListOrderLandmark();

    @Query("SELECT COUNT(*) FROM recognized_images")
    int getCount();

    @Query("SELECT COUNT(*) FROM recognized_images WHERE path = :myPath")
    int getCountByPath(String myPath);

    @Query("SELECT country, COUNT(*) AS count FROM recognized_images GROUP BY country")
    List<CountryCount> getCountries();

    @Query("SELECT locality, COUNT(*) AS count FROM recognized_images GROUP BY locality")
    List<LocalityCount> getLocalities();

    @Query("SELECT landmark_name, COUNT(*) AS count FROM recognized_images GROUP BY landmark_name")
    List<LandmarkCount> getLandmarks();

    @Query("SELECT * FROM recognized_images WHERE path = :myPath")
    RecognizedImages getImageByPath(String myPath);

    @Query("SELECT * FROM recognized_images WHERE country = :myCountry")
    List<RecognizedImages> getImagesByCountry(String myCountry);

    @Query("SELECT * FROM recognized_images WHERE locality = :myLocality")
    List<RecognizedImages> getImagesByLocality(String myLocality);

    @Query("SELECT * FROM recognized_images WHERE landmark_name = :myLandmarkName")
    List<RecognizedImages> getImagesByLandmark(String myLandmarkName);

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    void insertRecognizedImages(RecognizedImages recognizedImages);

    @Update
    void updateRecognizedImages(RecognizedImages recognizedImages);

    @Delete
    void deleteRecognizedImages(RecognizedImages recognizedImages);
}
