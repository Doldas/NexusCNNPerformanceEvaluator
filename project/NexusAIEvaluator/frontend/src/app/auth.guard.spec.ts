// auth.guard.spec.ts
import { TestBed, async, inject } from '@angular/core/testing';
import { RouterTestingModule } from '@angular/router/testing';
import { Router, ActivatedRouteSnapshot, RouterStateSnapshot } from '@angular/router';
import { AuthGuard } from './auth.guard';
import { AuthService } from './auth.service';
import { Observable, of } from 'rxjs';

class MockAuthService {
  getToken() {
    // Implement your mock logic for getToken if needed
  }

  isTokenExpired(): Observable<any> {
    // Implement your mock logic for isTokenExpired if needed
    return of({ isExpired: false });
  }

  isLoggedIn: boolean = true; // You may need to adjust this based on your AuthService implementation

  logout() {
    // Implement your mock logic for logout if needed
  }
}

describe('AuthGuard', () => {
  let authGuard: AuthGuard;
  let authService: AuthService;
  let router: Router;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [RouterTestingModule],
      providers: [
        AuthGuard,
        { provide: AuthService, useClass: MockAuthService },
      ],
    });

    authGuard = TestBed.inject(AuthGuard);
    authService = TestBed.inject(AuthService);
    router = TestBed.inject(Router);
  });

  it('should be created', () => {
    expect(authGuard).toBeTruthy();
  });

  it('should allow activation when token is valid', async(
    inject([AuthGuard], (guard: AuthGuard) => {
      spyOn(authService, 'getToken').and.returnValue('validToken');
      spyOn(authService, 'isTokenExpired').and.returnValue(of({ isExpired: false }));

      guard.canActivate({} as ActivatedRouteSnapshot, {} as RouterStateSnapshot).subscribe((result) => {
        expect(result).toBe(true);
      });
    })
  ));

  it('should redirect to login page and return false when token is expired', async(
    inject([AuthGuard], (guard: AuthGuard) => {
      spyOn(authService, 'getToken').and.returnValue('expiredToken');
      spyOn(authService, 'isTokenExpired').and.returnValue(of({ isExpired: true }));
      spyOn(authService, 'logout');

      const navigateSpy = spyOn(router, 'navigate');

      guard.canActivate({} as ActivatedRouteSnapshot, {} as RouterStateSnapshot).subscribe((result) => {
        expect(result).toBe(false);
        expect(navigateSpy).toHaveBeenCalledWith(['/login']);
        expect(authService.logout).toHaveBeenCalled();
      });
    })
  ));

  it('should redirect to login page and return false when token is not present', async(
    inject([AuthGuard], (guard: AuthGuard) => {
      spyOn(authService, 'getToken').and.returnValue(null);
      spyOn(authService, 'isTokenExpired'); // Mock isTokenExpired not to be called
      spyOn(router, 'navigate');

      guard.canActivate({} as ActivatedRouteSnapshot, {} as RouterStateSnapshot).subscribe((result) => {
        expect(result).toBe(false);
        expect(router.navigate).toHaveBeenCalledWith(['/login']);
      });
    })
  ));
});
