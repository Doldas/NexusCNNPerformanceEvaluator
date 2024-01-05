import { Injectable,Inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { DOCUMENT } from "@angular/common";
@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private apiUrl = 'http://localhost:4000/users'; // Replace with your actual backend API URL
  private readonly TOKEN_KEY = 'authToken';
  isLoggedIn = false;

  constructor(private http: HttpClient,@Inject(DOCUMENT) private document: Document) {
    this.isLoggedIn = !!localStorage.getItem(this.TOKEN_KEY);
  }

  login(username: string, password: string): Observable<any> {
    const loginData = { username, password };
    return this.http.post<any>(`${this.apiUrl}/login`, loginData);
  }

  isTokenExpired(): Observable<any>{
      return this.http.get<any>(`${this.apiUrl}/isTokenExpired`);
  }

  logout(): void {
    localStorage.removeItem(this.TOKEN_KEY);
    this.isLoggedIn = false;
  }

  setToken(token: string): void {
    localStorage.setItem(this.TOKEN_KEY, token);
    this.isLoggedIn = true;
  }

  getToken(): string | null {
    return localStorage.getItem(this.TOKEN_KEY);
  }

}
