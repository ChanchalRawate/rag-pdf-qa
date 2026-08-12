package com.chanchal.rag_backend.client;

import com.chanchal.rag_backend.dto.QueryRequest;
import com.chanchal.rag_backend.dto.QueryResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.multipart.MultipartFile;
@Component
public class FastApiClient {

    private final RestClient restClient;

    @Value("${fastapi.url}")
    private String fastApiUrl;

    public FastApiClient(RestClient restClient) {
        this.restClient = restClient;
    }

    public QueryResponse askQuestion(QueryRequest request) {

        return restClient.post()
                .uri(fastApiUrl + "/query")
                .body(request)
                .retrieve()
                .body(QueryResponse.class);

    }
    public String uploadPdf(MultipartFile file) {

    try {

        ByteArrayResource resource = new ByteArrayResource(file.getBytes()) {
            @Override
            public String getFilename() {
                return file.getOriginalFilename();
            }
        };

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("pdf", resource);

        return restClient.post()
                .uri(fastApiUrl + "/upload-pdf")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(body)
                .retrieve()
                .body(String.class);

    } catch (Exception e) {
        throw new RuntimeException("Failed to upload PDF", e);
    }
}
}