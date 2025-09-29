// dashboard.js
document.addEventListener('DOMContentLoaded', function() {
    // Загружаем данные с бэкенда
    loadDashboardData();
    
    // Initialize charts with animations
    setTimeout(() => {
        initClassificationChart();
    }, 500);

    setTimeout(() => {
        initMetricsChart();
    }, 700);

    setTimeout(() => {
        initTimelineChart();
    }, 900);

    setTimeout(() => {
        initConfusionMatrix();
    }, 1100);

    // Add event listeners for action buttons
    document.querySelectorAll('.action-btn').forEach(button => {
        button.addEventListener('click', function() {
            const reviewItem = this.closest('.review-item');
            if (this.classList.contains('confirm')) {
                reviewItem.style.opacity = '0.7';
                reviewItem.style.transform = 'scale(0.98)';
                this.textContent = 'Confirmed ✓';
                this.style.background = 'linear-gradient(135deg, #10b981, #059669)';
                this.disabled = true;
                if (reviewItem.querySelector('.dismiss')) {
                    reviewItem.querySelector('.dismiss').disabled = true;
                }

                // Show confirmation animation
                showNotification('Review confirmed as inappropriate', 'success');
            } else if (this.classList.contains('dismiss')) {
                reviewItem.style.opacity = '0.7';
                reviewItem.style.transform = 'scale(0.98)';
                this.textContent = 'Dismissed ✓';
                this.style.background = 'linear-gradient(135deg, #6b7280, #4b5563)';
                this.disabled = true;
                if (reviewItem.querySelector('.confirm')) {
                    reviewItem.querySelector('.confirm').disabled = true;
                }

                // Show dismissal animation
                showNotification('Review dismissed as clean', 'info');
            }
        });
    });

    // Add hover effects to cards
    document.querySelectorAll('.overview-card, .chart-card, .details-card').forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-8px) scale(1.02)';
        });

        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });
});

// Функция загрузки данных дашборда
async function loadDashboardData() {
    try {
        // Загружаем статистику обработанных файлов
        const processedFilesResponse = await fetch('http://localhost:8000/dashboard/processed-files');
        const processedFilesStats = await processedFilesResponse.json();
        
        // Обновляем статистику на странице
        updateDashboardStats(processedFilesStats);
        
        // Загружаем данные для графиков
        const chartsResponse = await fetch('http://localhost:8000/dashboard/charts');
        const chartsData = await chartsResponse.json();
        
        // Сохраняем данные для графиков
        window.dashboardChartsData = chartsData;
        
        // Загружаем недавние отзывы
        const reviewsResponse = await fetch('http://localhost:8000/dashboard/recent-reviews');
        const recentReviews = await reviewsResponse.json();
        
        // Обновляем список отзывов
        updateRecentReviews(recentReviews);
        
        // Показываем уведомление об успешной загрузке
        if (processedFilesStats.total_files > 0) {
            showNotification(`Загружена статистика по ${processedFilesStats.total_files} файлам`, 'success');
        } else {
            showNotification('Нет обработанных файлов. Загрузите CSV для анализа.', 'info');
        }
        
    } catch (error) {
        console.error('Ошибка загрузки данных дашборда:', error);
        // Показываем уведомление об ошибке
        showNotification('Ошибка загрузки данных дашборда', 'error');
    }
}

// Функция обновления статистики
function updateDashboardStats(stats) {
    // Обновляем метрики производительности модели
    const performanceStats = document.querySelectorAll('.overview-card:first-child .overview-stats .stat');
    if (performanceStats.length >= 3) {
        performanceStats[0].querySelector('.stat-value').textContent = stats.accuracy + '%';
        performanceStats[1].querySelector('.stat-value').textContent = stats.f1_score + '%';
        performanceStats[2].querySelector('.stat-value').textContent = stats.precision + '%';
    }
    
    // Обновляем статистику отзывов
    const reviewStats = document.querySelectorAll('.overview-card:last-child .overview-stats .stat');
    if (reviewStats.length >= 3) {
        reviewStats[0].querySelector('.stat-value').textContent = stats.total_reviews.toLocaleString();
        reviewStats[1].querySelector('.stat-value').textContent = stats.flagged_reviews.toLocaleString();
        reviewStats[2].querySelector('.stat-value').textContent = stats.flag_rate + '%';
    }
    
    // Обновляем информацию о модели
    const modelInfo = document.querySelector('.model-info');
    if (modelInfo) {
        const infoItems = modelInfo.querySelectorAll('.info-item');
        if (infoItems.length >= 5) {
            // Response Time
            if (stats.total_processing_time > 0) {
                const avgTime = (stats.total_processing_time / stats.total_reviews * 1000).toFixed(0);
                infoItems[4].querySelector('.info-value').textContent = `~${avgTime}ms per review`;
            } else {
                infoItems[4].querySelector('.info-value').textContent = '~120ms per review';
            }
        }
    }
    
    // Обновляем заголовок дашборда
    const dashboardHeader = document.querySelector('.dashboard-header h2');
    if (dashboardHeader && stats.total_files > 0) {
        dashboardHeader.textContent = `Review Moderation Dashboard (${stats.total_files} files processed)`;
    }
    
    console.log('Статистика обновлена:', stats);
}

// Функция обновления недавних отзывов
function updateRecentReviews(reviews) {
    const reviewsList = document.querySelector('.reviews-list');
    if (!reviewsList) return;
    
    reviewsList.innerHTML = '';
    
    reviews.forEach(review => {
        const reviewItem = document.createElement('div');
        reviewItem.className = `review-item ${review.flagged ? 'flagged' : 'clean'}`;
        
        if (review.is_file_info) {
            // Показываем информацию о файле
            reviewItem.innerHTML = `
                <div class="review-content">
                    <p><strong>${review.text}</strong></p>
                    <p>Токсичных: ${review.flagged_reviews}, Чистых: ${review.clean_reviews}</p>
                    <span class="review-date">${review.date}</span>
                </div>
                <div class="review-actions">
                    <span class="status-clean">Файл обработан</span>
                </div>
            `;
        } else {
            // Показываем обычный отзыв
            reviewItem.innerHTML = `
                <div class="review-content">
                    <p>"${review.text}"</p>
                    <span class="review-date">${review.date}</span>
                </div>
                <div class="review-actions">
                    ${review.flagged ? `
                        <button class="action-btn confirm" data-review-id="${review.id}">Confirm</button>
                        <button class="action-btn dismiss" data-review-id="${review.id}">Dismiss</button>
                    ` : `
                        <span class="status-clean">Clean</span>
                    `}
                </div>
            `;
        }
        
        reviewsList.appendChild(reviewItem);
    });
    
    // Добавляем обработчики событий для новых кнопок
    document.querySelectorAll('.action-btn').forEach(button => {
        button.addEventListener('click', function() {
            const reviewId = this.getAttribute('data-review-id');
            const action = this.classList.contains('confirm') ? 'confirm' : 'dismiss';
            handleReviewAction(reviewId, action, this);
        });
    });
}

// Функция обработки действий с отзывами
async function handleReviewAction(reviewId, action, buttonElement) {
    try {
        const response = await fetch(`http://localhost:8000/dashboard/review/${reviewId}/action`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ action: action })
        });
        
        if (response.ok) {
            const reviewItem = buttonElement.closest('.review-item');
            if (buttonElement.classList.contains('confirm')) {
                reviewItem.style.opacity = '0.7';
                reviewItem.style.transform = 'scale(0.98)';
                buttonElement.textContent = 'Confirmed ✓';
                buttonElement.style.background = 'linear-gradient(135deg, #10b981, #059669)';
                buttonElement.disabled = true;
                if (reviewItem.querySelector('.dismiss')) {
                    reviewItem.querySelector('.dismiss').disabled = true;
                }
                showNotification('Review confirmed as inappropriate', 'success');
            } else if (buttonElement.classList.contains('dismiss')) {
                reviewItem.style.opacity = '0.7';
                reviewItem.style.transform = 'scale(0.98)';
                buttonElement.textContent = 'Dismissed ✓';
                buttonElement.style.background = 'linear-gradient(135deg, #6b7280, #4b5563)';
                buttonElement.disabled = true;
                if (reviewItem.querySelector('.confirm')) {
                    reviewItem.querySelector('.confirm').disabled = true;
                }
                showNotification('Review dismissed as clean', 'info');
            }
        } else {
            throw new Error('Ошибка обработки действия');
        }
    } catch (error) {
        console.error('Ошибка обработки действия с отзывом:', error);
        showNotification('Ошибка обработки действия', 'error');
    }
}

function initClassificationChart() {
    const ctx = document.getElementById('classificationChart').getContext('2d');

    // Add animation delay
    Chart.defaults.animation.delay = 300;

    // Используем данные с бэкенда или заглушку
    const chartData = window.dashboardChartsData?.classification || {
        labels: ['Clean Reviews', 'Flagged Reviews'],
        data: [14474, 1254]
    };

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: chartData.labels,
            datasets: [{
                data: chartData.data,
                backgroundColor: [
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(239, 68, 68, 0.8)'
                ],
                borderColor: [
                    'rgba(16, 185, 129, 1)',
                    'rgba(239, 68, 68, 1)'
                ],
                borderWidth: 2,
                hoverOffset: 15
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 25,
                        usePointStyle: true,
                        font: {
                            size: 14,
                            family: "'Mulish', sans-serif"
                        },
                        color: '#fff'
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.7)',
                    titleFont: {
                        family: "'Mulish', sans-serif"
                    },
                    bodyFont: {
                        family: "'Mulish', sans-serif"
                    },
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.raw;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = Math.round((value / total) * 100);
                            return `${label}: ${value.toLocaleString()} (${percentage}%)`;
                        }
                    }
                }
            },
            animation: {
                animateScale: true,
                animateRotate: true
            }
        }
    });
}

function initMetricsChart() {
    const ctx = document.getElementById('metricsChart').getContext('2d');

    // Используем данные с бэкенда или заглушку
    const chartData = window.dashboardChartsData?.metrics || {
        labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        data: [94.2, 93.5, 92.1, 92.8]
    };

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: chartData.labels,
            datasets: [{
                label: 'Performance Metrics',
                data: chartData.data,
                backgroundColor: [
                    'rgba(139, 92, 246, 0.7)',
                    'rgba(99, 102, 241, 0.7)',
                    'rgba(79, 70, 229, 0.7)',
                    'rgba(67, 56, 202, 0.7)'
                ],
                borderColor: [
                    'rgb(139, 92, 246)',
                    'rgb(99, 102, 241)',
                    'rgb(79, 70, 229)',
                    'rgb(67, 56, 202)'
                ],
                borderWidth: 1,
                borderRadius: 8,
                borderSkipped: false,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        },
                        color: '#ccc',
                        font: {
                            family: "'Mulish', sans-serif"
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                x: {
                    ticks: {
                        color: '#ccc',
                        font: {
                            family: "'Mulish', sans-serif",
                            size: 12
                        }
                    },
                    grid: {
                        display: false
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.7)',
                    titleFont: {
                        family: "'Mulish', sans-serif"
                    },
                    bodyFont: {
                        family: "'Mulish', sans-serif"
                    },
                    callbacks: {
                        label: function(context) {
                            return context.parsed.y + '%';
                        }
                    }
                }
            },
            animation: {
                duration: 2000,
                easing: 'easeOutQuart'
            }
        }
    });
}

function initTimelineChart() {
    const ctx = document.getElementById('timelineChart').getContext('2d');

    // Используем данные с бэкенда или заглушку
    const chartData = window.dashboardChartsData?.timeline || {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct'],
        flagged: [85, 92, 78, 105, 120, 98, 134, 110, 125, 142],
        clean: [1120, 1245, 1320, 1450, 1520, 1680, 1750, 1820, 1940, 2010]
    };

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.labels,
            datasets: [{
                label: 'Flagged Reviews',
                data: chartData.flagged,
                borderColor: 'rgb(239, 68, 68)',
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                tension: 0.3,
                fill: true,
                pointBackgroundColor: 'rgb(239, 68, 68)',
                pointBorderColor: '#fff',
                pointBorderWidth: 2,
                pointRadius: 5,
                pointHoverRadius: 7
            }, {
                label: 'Clean Reviews',
                data: chartData.clean,
                borderColor: 'rgb(16, 185, 129)',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                tension: 0.3,
                fill: true,
                pointBackgroundColor: 'rgb(16, 185, 129)',
                pointBorderColor: '#fff',
                pointBorderWidth: 2,
                pointRadius: 5,
                pointHoverRadius: 7
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        color: '#ccc',
                        font: {
                            family: "'Mulish', sans-serif"
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                x: {
                    ticks: {
                        color: '#ccc',
                        font: {
                            family: "'Mulish', sans-serif"
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#fff',
                        font: {
                            family: "'Mulish', sans-serif",
                            size: 12
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.7)',
                    titleFont: {
                        family: "'Mulish', sans-serif"
                    },
                    bodyFont: {
                        family: "'Mulish', sans-serif"
                    }
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            },
            animation: {
                duration: 2000,
                easing: 'easeOutQuart'
            }
        }
    });
}

function initConfusionMatrix() {
    const ctx = document.getElementById('confusionMatrix').getContext('2d');

    const data = {
        labels: ['True Negative', 'False Positive', 'False Negative', 'True Positive'],
        datasets: [{
            label: 'Count',
            data: [13520, 954, 98, 1156],
            backgroundColor: [
                'rgba(16, 185, 129, 0.7)',
                'rgba(239, 68, 68, 0.7)',
                'rgba(239, 68, 68, 0.7)',
                'rgba(16, 185, 129, 0.7)'
            ],
            borderColor: [
                'rgb(16, 185, 129)',
                'rgb(239, 68, 68)',
                'rgb(239, 68, 68)',
                'rgb(16, 185, 129)'
            ],
            borderWidth: 2,
            borderRadius: 8,
            borderSkipped: false,
        }]
    };

    new Chart(ctx, {
        type: 'bar',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        color: '#ccc',
                        font: {
                            family: "'Mulish', sans-serif"
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                x: {
                    ticks: {
                        color: '#ccc',
                        font: {
                            family: "'Mulish', sans-serif",
                            size: 11
                        }
                    },
                    grid: {
                        display: false
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.7)',
                    titleFont: {
                        family: "'Mulish', sans-serif"
                    },
                    bodyFont: {
                        family: "'Mulish', sans-serif"
                    },
                    callbacks: {
                        title: function(tooltipItems) {
                            return tooltipItems[0].label;
                        },
                        label: function(context) {
                            return `Count: ${context.parsed.y.toLocaleString()}`;
                        }
                    }
                }
            },
            animation: {
                duration: 2000,
                easing: 'easeOutQuart'
            }
        }
    });
}

function showNotification(message, type) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <span>${message}</span>
        <button onclick="this.parentElement.remove()">×</button>
    `;

    // Add styles
    notification.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        background: ${type === 'success' ? 'linear-gradient(135deg, #10b981, #059669)' : 'linear-gradient(135deg, #3b82f6, #2563eb)'};
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        z-index: 10000;
        display: flex;
        align-items: center;
        gap: 15px;
        animation: slideInRight 0.3s ease;
        max-width: 300px;
    `;

    // Add button styles
    notification.querySelector('button').style.cssText = `
        background: none;
        border: none;
        color: white;
        font-size: 20px;
        cursor: pointer;
        padding: 0;
        width: 20px;
        height: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
    `;

    document.body.appendChild(notification);

    // Auto remove after 3 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => {
                if (notification.parentElement) {
                    notification.remove();
                }
            }, 300);
        }
    }, 3000);
}

// Add CSS for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOutRight {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);