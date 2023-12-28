// auth-interceptor.service.ts
import { Injectable } from '@angular/core';
import {
  HttpInterceptor,
  HttpRequest,
  HttpHandler,
  HttpEvent,
} from '@angular/common/http';
import { Observable } from 'rxjs';
import { AuthService } from './auth.service'; // Make sure to import your AuthService

@Injectable()
export class AuthInterceptor implements HttpInterceptor {
  constructor(private authService: AuthService) {}

  intercept(
    request: HttpRequest<any>,
    next: HttpHandler
  ): Observable<HttpEvent<any>> {
    // Get the authentication token from the service
    const authToken = this.authService.getToken();

    // Clone the request and add the Authorization header
    const authRequest = authToken
      ? request.clone({
          setHeaders: { Authorization: `Bearer ${authToken}` },
        })
      : request;

    // Pass the cloned request with the new header to the next handler
    return next.handle(authRequest);
  }
}
