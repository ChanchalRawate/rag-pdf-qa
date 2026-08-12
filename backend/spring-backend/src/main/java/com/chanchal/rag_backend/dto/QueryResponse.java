package com.chanchal.rag_backend.dto;

public class QueryResponse {

    private boolean success;
    private String answer;

    public QueryResponse() {}

    public QueryResponse(boolean success, String answer) {
        this.success = success;
        this.answer = answer;
    }

    public boolean isSuccess() {
        return success;
    }

    public void setSuccess(boolean success) {
        this.success = success;
    }

    public String getAnswer() {
        return answer;
    }

    public void setAnswer(String answer) {
        this.answer = answer;
    }
}