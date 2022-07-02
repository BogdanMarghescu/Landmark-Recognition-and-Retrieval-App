package ro.mta.landmarkrecognitionapp;

import android.content.Context;

import java.util.concurrent.TimeUnit;

import okhttp3.OkHttpClient;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class NetworkClient {
    private static Retrofit retrofit;

    public static Retrofit getRetrofit(Context context) {
        OkHttpClient okHttpClient = new OkHttpClient.Builder().
                readTimeout(60, TimeUnit.SECONDS).
                connectTimeout(20, TimeUnit.SECONDS).
                build();
        if (retrofit == null) {
            retrofit = new Retrofit.Builder().baseUrl(context.getResources().
                    getString(R.string.base_url_gpu_server)).
                    addConverterFactory(GsonConverterFactory.create()).
                    client(okHttpClient).
                    build();
        }
        return retrofit;
    }
}
