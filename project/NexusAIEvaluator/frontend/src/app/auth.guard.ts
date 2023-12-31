// auth.guard.ts
import { Injectable } from '@angular/core';
import { CanActivate, ActivatedRouteSnapshot, RouterStateSnapshot, Router } from '@angular/router';
import { AuthService } from './auth.service';
import { map } from 'rxjs/operators';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class AuthGuard implements CanActivate {
  constructor(private authService: AuthService, private router: Router) {}

  canActivate(route: ActivatedRouteSnapshot, state: RouterStateSnapshot): boolean | Observable<boolean> {
    const token = this.authService.getToken();
    if(token != null && token != ""){
      return this.authService.isTokenExpired().pipe(map((response)=>{
        if(response.isExpired){
          this.authService.logout();
          // Redirect to the login page if not logged in
          this.router.navigate(['/login']);
          return false;
        }
        return this.authService.isLoggedIn;
      }));
    }
    this.router.navigate(['/login']);
    return false;
  }
}
