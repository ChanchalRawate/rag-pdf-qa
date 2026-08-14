package com.chanchal.rag_backend.security;

import io.jsonwebtoken.Claims;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.io.Decoders;
import io.jsonwebtoken.security.Keys;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.stereotype.Service;

import javax.crypto.SecretKey;
import java.util.Date;
import java.util.function.Function;

@Service
public class JwtService {

    @Value("${jwt.secret}")
    private String secretKey;

    // =========================================================
    // Get signing key
    // =========================================================

    private SecretKey getSigningKey() {

        byte[] keyBytes = Decoders.BASE64.decode(secretKey);

        return Keys.hmacShaKeyFor(keyBytes);
    }

    // =========================================================
    // Generate JWT
    // =========================================================

    public String generateToken(String username) {

        return Jwts.builder()
                .subject(username)
                .issuedAt(new Date())
                .expiration(
                        new Date(System.currentTimeMillis() + 60 * 60 * 1000)
                )
                .signWith(getSigningKey())
                .compact();
    }

    // =========================================================
    // Extract username from JWT
    // =========================================================

    public String extractUsername(String token) {

        return extractClaim(token, Claims::getSubject);
    }

    // =========================================================
    // Extract any claim
    // =========================================================

    public <T> T extractClaim(
            String token,
            Function<Claims, T> claimsResolver
    ) {

        final Claims claims = extractAllClaims(token);

        return claimsResolver.apply(claims);
    }

    // =========================================================
    // Extract all claims and verify JWT signature
    // =========================================================

    private Claims extractAllClaims(String token) {

        return Jwts.parser()
                .verifyWith(getSigningKey())
                .build()
                .parseSignedClaims(token)
                .getPayload();
    }

    // =========================================================
    // Check whether token is valid
    // =========================================================

    public boolean isTokenValid(
            String token,
            UserDetails userDetails
    ) {

        final String username = extractUsername(token);

        return username.equals(userDetails.getUsername())
                && !isTokenExpired(token);
    }

    // =========================================================
    // Check expiration
    // =========================================================

    private boolean isTokenExpired(String token) {

        return extractExpiration(token).before(new Date());
    }

    // =========================================================
    // Extract expiration date
    // =========================================================

    private Date extractExpiration(String token) {

        return extractClaim(token, Claims::getExpiration);
    }
}