package com.chanchal.rag_backend.dto;

public class QueryRequest {

    private String question;

    public QueryRequest() {
    }

    public String getQuestion() {
        return question;
    }

    public void setQuestion(String question) {
        this.question = question;
    }
}