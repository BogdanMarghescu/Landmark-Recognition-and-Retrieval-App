package ro.mta.landmarkrecognitionapp;

import android.content.Context;

import androidx.annotation.NonNull;
import androidx.room.Database;
import androidx.room.DatabaseConfiguration;
import androidx.room.InvalidationTracker;
import androidx.room.Room;
import androidx.room.RoomDatabase;
import androidx.sqlite.db.SupportSQLiteOpenHelper;

@Database(entities = {RecognizedImages.class, UnrecognizedImages.class}, version = 1)
public abstract class LandmarkRecognitionDatabase extends RoomDatabase {
    private static LandmarkRecognitionDatabase instance;

    public abstract RecognizedImagesDao recognizedImagesDao();
    public abstract UnrecognizedImagesDao unrecognizedImagesDao();

    public static LandmarkRecognitionDatabase getInstance(Context context) {
        if (instance == null) {
            instance = Room.databaseBuilder(context.getApplicationContext(), LandmarkRecognitionDatabase.class, "landmark_recognition_db").allowMainThreadQueries().build();
        }
        return instance;
    }

    @NonNull
    @Override
    protected SupportSQLiteOpenHelper createOpenHelper(DatabaseConfiguration config) {
        return null;
    }

    @NonNull
    @Override
    protected InvalidationTracker createInvalidationTracker() {
        return null;
    }

    @Override
    public void clearAllTables() {

    }
}
