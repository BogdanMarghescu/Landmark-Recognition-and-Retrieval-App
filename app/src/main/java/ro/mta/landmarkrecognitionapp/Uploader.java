package ro.mta.landmarkrecognitionapp;

import okhttp3.MultipartBody;
import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.http.Multipart;
import retrofit2.http.POST;
import retrofit2.http.Part;

public interface Uploader {
    @Multipart
    @POST("detect")
    Call<ResponseBody> upload(@Part MultipartBody.Part file);
}
