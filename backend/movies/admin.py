from django.contrib import admin
from .models import Movie, ChatSession


@admin.register(Movie)
class MovieAdmin(admin.ModelAdmin):
    list_display = ("serial_name", "release_date", "vote_average", "popularity")
    list_filter = ("original_language",)
    search_fields = ("serial_name", "original_title", "description")


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ("session_id", "turn_count", "created_at", "updated_at")
    readonly_fields = ("session_id", "created_at", "updated_at")
