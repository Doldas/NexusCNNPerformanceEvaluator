import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { AuthService } from '../auth.service';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent {
  loginForm: FormGroup;
  errorMessage = '';

  constructor(private authService: AuthService, private router: Router,private fb: FormBuilder) {
    this.loginForm = this.fb.group({
      username: ['', Validators.required],
      password: ['', Validators.required],
    });
  }

  login(): void {
    if (this.loginForm.valid) {
      // Proceed with login logic
      this.errorMessage = ''; // Clear any previous error messages
      this.authService.login(this.loginForm.get('username')?.value, this.loginForm.get('password')?.value).subscribe(
        (response) => {
          const jwtToken = response.token;

          if (jwtToken) {
            this.authService.setToken(jwtToken);
            // Navigate to the home page after successful login
            this.router.navigate(['/home']);
          } else {
            this.errorMessage = 'Invalid credentials. Please try again.';
          }
        },
        (error) => {
          this.errorMessage = 'Invalid credentials. Please try again.';
        }
      );
    } else {
      // Display error message
      this.errorMessage = 'Please enter valid credentials.';
    }
  }

  logout(): void {
    this.authService.logout();
  }

  isLoggedIn(): Boolean {
    return this.authService.isLoggedIn;
  }
}
