package com.chanchal.rag_backend.controller;

import com.chanchal.rag_backend.dto.QueryRequest;
import com.chanchal.rag_backend.service.QueryService;
import org.springframework.web.bind.annotation.*;
import com.chanchal.rag_backend.dto.QueryResponse;
import org.springframework.web.multipart.MultipartFile;

@RestController
@CrossOrigin(origins = "http://localhost:5173")
public class HomeController {

    private final QueryService queryService;

    public HomeController(QueryService queryService) {
        this.queryService = queryService;
    }

    @GetMapping("/")
    public String home() {
        return "Spring Boot Backend Running!";
    }

    @PostMapping("/query")
    public QueryResponse query(@RequestBody QueryRequest request) {
        return queryService.processQuestion(request.getQuestion());
    }

    @PostMapping("/upload-pdf")
    public String uploadPdf(@RequestParam("pdf") MultipartFile file) {

        return queryService.uploadPdf(file);

    }
}
