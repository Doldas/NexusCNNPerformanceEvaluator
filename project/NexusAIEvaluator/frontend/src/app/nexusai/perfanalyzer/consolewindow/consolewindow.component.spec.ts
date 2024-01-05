import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ConsolewindowComponent } from './consolewindow.component';

describe('ConsolewindowComponent', () => {
  let component: ConsolewindowComponent;
  let fixture: ComponentFixture<ConsolewindowComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ConsolewindowComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(ConsolewindowComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
