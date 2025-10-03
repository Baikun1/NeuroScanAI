from django.contrib.auth.models import AbstractUser, Group, Permission, UserManager
from django.db import models
from django.core.mail import EmailMessage

class CustomUserManager(UserManager):
    def create_superuser(self, email=None, name=None, phone=None, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        if not password:
            raise ValueError("Superuser must have a password.")
        return self._create_user(email, email, password, name=name, phone=phone, **extra_fields)

class EmailTemplate(models.Model):
    TEMPLATE_TYPES = [
        ('no_tumor', 'No Tumor'),
        ('tumor', 'Tumor'),
    ]
    template_type = models.CharField(max_length=20, choices=TEMPLATE_TYPES, unique=True)
    subject = models.CharField(max_length=255)
    body = models.TextField()

    def __str__(self):
        return self.template_type

    @staticmethod
    def load_templates():
        import os
        base_path = os.path.join('mri_app', 'templates', 'mri_app')
        no_tumor_path = os.path.join(base_path, 'email_no_tumor.html')
        tumor_path = os.path.join(base_path, 'email_tumor.html')

        with open(no_tumor_path, 'r') as f:
            no_tumor_body = f.read()
        with open(tumor_path, 'r') as f:
            tumor_body = f.read()

        EmailTemplate.objects.update_or_create(
            template_type='no_tumor',
            defaults={'subject': 'MRI Scan Result: No Tumor Detected', 'body': no_tumor_body}
        )
        EmailTemplate.objects.update_or_create(
            template_type='tumor',
            defaults={'subject': 'MRI Scan Result: Tumor Detected', 'body': tumor_body}
        )

class User(AbstractUser):
    phone = models.CharField(max_length=15)
    mri_image = models.ImageField(upload_to='mri_images/')
    diagnosis = models.CharField(max_length=100, blank=True)
    name = models.CharField(max_length=100)

    email = models.EmailField(unique=True)

    groups = models.ManyToManyField(Group, related_name='mri_users', blank=True)
    user_permissions = models.ManyToManyField(Permission, related_name='mri_users', blank=True)

    objects = CustomUserManager()

    class Meta:
        unique_together = ('phone', 'email')

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['name', 'phone']

    def __str__(self):
        return self.name

    def send_result_email(self, result):
        import os
        import logging
        from django.template import Template, Context
        from django.conf import settings
        from django.core.mail import EmailMessage

        logger = logging.getLogger(__name__)

        # Ensure templates are loaded
        EmailTemplate.load_templates()

        template_type = 'no_tumor' if result.get('predicted_class') == 'notumor' else 'tumor'
        try:
            template_obj = EmailTemplate.objects.get(template_type=template_type)
            subject = template_obj.subject
            template_body = template_obj.body

            # Render the HTML template with context
            template = Template(template_body)
            context = Context({
                'name': self.name,
                'phone': self.phone,
                'email': self.email,
                'predicted_class': result['predicted_class'],
                'probabilities': result['probabilities'],
            })
            body_html = template.render(context)

            email = EmailMessage(
                subject=subject,
                body=body_html,
                from_email=settings.DEFAULT_FROM_EMAIL,
                to=[self.email],
            )
            email.content_subtype = "html"  # Main content is now text/html

            # Attach MRI image file if exists
            if self.mri_image and os.path.exists(self.mri_image.path):
                email.attach_file(self.mri_image.path)

            email.send()
            return True
        except EmailTemplate.DoesNotExist:
            logger.error(f"Email template for {template_type} not found.")
            return False
        except Exception as e:
            logger.error(f"Failed to send email to {self.email}: {str(e)}")
            return False
