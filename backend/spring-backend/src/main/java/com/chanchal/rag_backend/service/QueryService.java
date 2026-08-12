package com.chanchal.rag_backend.service;

import com.chanchal.rag_backend.client.FastApiClient;
import com.chanchal.rag_backend.dto.QueryRequest;
import com.chanchal.rag_backend.dto.QueryResponse;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

@Service
public class QueryService {

    private final FastApiClient fastApiClient;

    public QueryService(FastApiClient fastApiClient) {
        this.fastApiClient = fastApiClient;
    }

    public QueryResponse processQuestion(String question) {

        QueryRequest request = new QueryRequest();
        request.setQuestion(question);

        return fastApiClient.askQuestion(request);
    }

    public String uploadPdf(MultipartFile file) {

        return fastApiClient.uploadPdf(file);

    }
}