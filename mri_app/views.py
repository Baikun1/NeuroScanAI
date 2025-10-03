import os
from django.shortcuts import render, redirect
from django.contrib import messages
from django.views import View
from .models import User
from .model_train.predict import MRIPredictor

# Instantiate the predictor once to avoid reloading the model on each request
predictor = MRIPredictor()

class IndexView(View):
    def get(self, request):
        return render(request, 'mri_app/index1.html')

    def post(self, request):
        try:
            name = request.POST.get('name')
            phone = request.POST.get('phone')
            email = request.POST.get('email')
            mri_image = request.FILES.get('mri_image')

            if not all([name, phone, email, mri_image]):
                messages.error(request, 'All fields are required.')
                return redirect('index')

            # Validate file size (10MB limit)
            if mri_image.size > 10 * 1024 * 1024:
                messages.error(request, 'File size must be less than 10MB.')
                return redirect('index')

            # Validate file type
            allowed_types = ['image/jpeg', 'image/jpg', 'image/png']
            if mri_image.content_type not in allowed_types:
                messages.error(request, 'Only JPEG, JPG, and PNG image files are allowed.')
                return redirect('index')

            # Check if user exists by email only, update if exists, else create
            user = User.objects.filter(email=email).first()
            if user:
                updated = False
                if user.name != name:
                    user.name = name
                    updated = True
                if user.phone != phone:
                    user.phone = phone
                    updated = True
                # Delete old MRI image file if exists before updating
                if user.mri_image and user.mri_image.name and user.mri_image.name != mri_image.name:
                    old_image_path = user.mri_image.path
                    if os.path.exists(old_image_path):
                        os.remove(old_image_path)
                # Always update mri_image for new MRI check
                user.mri_image = mri_image
                updated = True
                if updated:
                    user.save()
            else:
                user = User.objects.create(name=name, phone=phone, email=email, mri_image=mri_image)

            # Predict
            img_path = user.mri_image.path
            result = predictor.predict(img_path)

            if result is None:
                messages.error(request, 'Failed to process the MRI image. Please ensure it is a valid brain MRI scan.')
                return redirect('index')

            # Save diagnosis to user
            user.diagnosis = result
            user.save()

            # Send email with result
            email_sent = user.send_result_email(result)
            if email_sent:
                messages.success(request, 'Email report sent to your email.')
            else:
                messages.error(request, 'Failed to send email report. Please check your email settings.')

            return render(request, 'mri_app/index1.html', {
                'result': result,
                'user': user
            })

        except Exception as e:
            messages.error(request, f'An error occurred while processing your request: {str(e)}. Please try again.')
            return redirect('index')

# For backward compatibility, keep function view if needed
index = IndexView.as_view()
