import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AiperfanalyzerformComponent } from './aiperfanalyzerform.component';

describe('AiperfanalyzerformComponent', () => {
  let component: AiperfanalyzerformComponent;
  let fixture: ComponentFixture<AiperfanalyzerformComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AiperfanalyzerformComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(AiperfanalyzerformComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
