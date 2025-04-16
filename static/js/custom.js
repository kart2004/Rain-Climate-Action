
(function($) {
    // 'use strict';

    // Main Navigation
    $( '.hamburger-menu' ).on( 'click', function() {
        $(this).toggleClass('open');
        $('.site-navigation').toggleClass('show');
    });

    // Hero Slider
    var mySwiper = new Swiper('.hero-slider', {
        slidesPerView: 1,
        spaceBetween: 0,
        loop: true,
        pagination: {
            el: '.swiper-pagination',
            clickable: true,
            renderBullet: function (index, className) {
                return '<span class="' + className + '">0' + (index + 1) + '</span>';
            },
        },
        navigation: {
            nextEl: '.swiper-button-next',
            prevEl: '.swiper-button-prev'
        }
    });

    // Cause Slider
    var causesSlider = new Swiper('.causes-slider', {
        slidesPerView: 3,
        spaceBetween: 30,
        navigation: {
            nextEl: '.swiper-button-next',
            prevEl: '.swiper-button-prev'
        },
        breakpoints: {
            1200: {
                slidesPerView: 2,
                spaceBetween: 30,
            },
            768: {
                slidesPerView: 1,
                spaceBetween: 0,
            }
        }
    } );

    // Accordion & Toggle
    $('.accordion-wrap.type-accordion').collapsible({
        accordion: true,
        contentOpen: 0,
        arrowRclass: 'arrow-r',
        arrowDclass: 'arrow-d'
    });

    $('.accordion-wrap .entry-title').on('click', function() {
        $('.accordion-wrap .entry-title').removeClass('active');
        $(this).addClass('active');
    });

    // Tabs
    $(function() {
        $('.tab-content:first-child').show();

        $('.tab-nav').bind('click', function(e) {
            $this = $(this);
            $tabs = $this.parent().parent().next();
            $target = $($this.data("target"));
            $this.siblings().removeClass('active');
            $target.siblings().css("display", "none");
            $this.addClass('active');
            $target.fadeIn("slow");
        });

        $('.tab-nav:first-child').trigger('click');
    });


    // Circular Progress Bar Severity 0
    $('#loader_1_0').circleProgress({
        startAngle: -Math.PI / 4 * 2,
        value: 0.02,
        size: 156,
        thickness: 3,
        fill: {
            gradient: ["#189ad3", "#189ad3"]
        }
    }).on('circle-animation-progress', function(event, progress) {
        $(this).find('strong').html(Math.round(0 * progress));
    });

    $('#loader_2_0').circleProgress({
        startAngle: -Math.PI / 4 * 2,
        value: 0.99999,
        size: 156,
        thickness: 3,
        fill: {
            gradient: ["#189ad3", "#189ad3"]
        }
    }).on('circle-animation-progress', function(event, progress) {
        $(this).find('strong').html(Math.round(100 * progress) + '<i>%</i>');
    });

    $('#loader_3_0').circleProgress({
        startAngle: -Math.PI / 4 * 2,
        value: 0.02,
        size: 156,
        thickness: 3,
        fill: {
            gradient: ["#189ad3", "#189ad3"]
        }
    }).on('circle-animation-progress', function(event, progress) {
        $(this).find('strong').html(Math.round(0 * progress) + '<i>ft</i>');
    });

    // Circular Progress Bar Severity 1
    $('#loader_1_1').circleProgress({
        startAngle: -Math.PI / 4 * 2,
        value: 0.20,
        size: 156,
        thickness: 3,
        fill: {
            gradient: ["#189ad3", "#189ad3"]
        }
    }).on('circle-animation-progress', function(event, progress) {
        $(this).find('strong').html(Math.round(1 * progress));
    });

    $('#loader_2_1').circleProgress({
        startAngle: -Math.PI / 4 * 2,
        value: 0.90,
        size: 156,
        thickness: 3,
        fill: {
            gradient: ["#189ad3", "#189ad3"]
        }
    }).on('circle-animation-progress', function(event, progress) {
        $(this).find('strong').html(Math.round(90 * progress) + '<i>%</i>');
    });

    $('#loader_3_1').circleProgress({
        startAngle: -Math.PI / 4 * 2,
        value: 0.10,
        size: 156,
        thickness: 3,
        fill: {
            gradient: ["#189ad3", "#189ad3"]
        }
    }).on('circle-animation-progress', function(event, progress) {
        $(this).find('strong').html(Math.round(1 * progress) + '<i>ft</i>');
    });

    // Circular Progress Bar Severity 2
    $('#loader_1_2').circleProgress({
        startAngle: -Math.PI / 4 * 2,
        value: 0.40,
        size: 156,
        thickness: 3,
        fill: {
            gradient: ["#189ad3", "#189ad3"]
        }
    }).on('circle-animation-progress', function(event, progress) {
        $(this).find('strong').html(Math.round(2 * progress));
    });

    $('#loader_2_2').circleProgress({
        startAngle: -Math.PI / 4 * 2,
        value: 0.80,
        size: 156,
        thickness: 3,
        fill: {
            gradient: ["#189ad3", "#189ad3"]
        }
    }).on('circle-animation-progress', function(event, progress) {
        $(this).find('strong').html(Math.round(80 * progress) + '<i>%</i>');
    });

    $('#loader_3_2').circleProgress({
        startAngle: -Math.PI / 4 * 2,
        value: 0.20,
        size: 156,
        thickness: 3,
        fill: {
            gradient: ["#189ad3", "#189ad3"]
        }
    }).on('circle-animation-progress', function(event, progress) {
        $(this).find('strong').html(Math.round(2 * progress) + '<i>ft</i>');
    });

    // Circular Progress Bar Severity 3
    $('#loader_1_3').circleProgress({
        startAngle: -Math.PI / 4 * 2,
        value: 0.60,
        size: 156,
        thickness: 3,
        fill: {
            gradient: ["#189ad3", "#189ad3"]
        }
    }).on('circle-animation-progress', function(event, progress) {
        $(this).find('strong').html(Math.round(3 * progress));
    });

    $('#loader_2_3').circleProgress({
        startAngle: -Math.PI / 4 * 2,
        value: 0.60,
        size: 156,
        thickness: 3,
        fill: {
            gradient: ["#189ad3", "#189ad3"]
        }
    }).on('circle-animation-progress', function(event, progress) {
        $(this).find('strong').html(Math.round(60 * progress) + '<i>%</i>');
    });

    $('#loader_3_3').circleProgress({
        startAngle: -Math.PI / 4 * 2,
        value: 0.50,
        size: 156,
        thickness: 3,
        fill: {
            gradient: ["#189ad3", "#189ad3"]
        }
    }).on('circle-animation-progress', function(event, progress) {
        $(this).find('strong').html(Math.round(5 * progress) + '<i>ft</i>');
    });

    // Circular Progress Bar Severity 4
    $('#loader_1_4').circleProgress({
        startAngle: -Math.PI / 4 * 2,
        value: 0.80,
        size: 156,
        thickness: 3,
        fill: {
            gradient: ["#189ad3", "#189ad3"]
        }
    }).on('circle-animation-progress', function(event, progress) {
        $(this).find('strong').html(Math.round(4 * progress));
    });

    $('#loader_2_4').circleProgress({
        startAngle: -Math.PI / 4 * 2,
        value: 0.40,
        size: 156,
        thickness: 3,
        fill: {
            gradient: ["#189ad3", "#189ad3"]
        }
    }).on('circle-animation-progress', function(event, progress) {
        $(this).find('strong').html(Math.round(40 * progress) + '<i>%</i>');
    });

    $('#loader_3_4').circleProgress({
        startAngle: -Math.PI / 4 * 2,
        value: 0.75,
        size: 156,
        thickness: 3,
        fill: {
            gradient: ["#189ad3", "#189ad3"]
        }
    }).on('circle-animation-progress', function(event, progress) {
        $(this).find('strong').html(Math.round(7.5 * progress) + '<i>ft</i>');
    });

    // Circular Progress Bar Severity 5
    $('#loader_1_5').circleProgress({
        startAngle: -Math.PI / 4 * 2,
        value: 0.9999,
        size: 156,
        thickness: 3,
        fill: {
            gradient: ["#189ad3", "#189ad3"]
        }
    }).on('circle-animation-progress', function(event, progress) {
        $(this).find('strong').html(Math.round(5 * progress));
    });

    $('#loader_2_5').circleProgress({
        startAngle: -Math.PI / 4 * 2,
        value: 0.20,
        size: 156,
        thickness: 3,
        fill: {
            gradient: ["#189ad3", "#189ad3"]
        }
    }).on('circle-animation-progress', function(event, progress) {
        $(this).find('strong').html(Math.round(20 * progress) + '<i>%</i>');
    });

    $('#loader_3_5').circleProgress({
        startAngle: -Math.PI / 4 * 2,
        value: 0.9999,
        size: 156,
        thickness: 3,
        fill: {
            gradient: ["#189ad3", "#189ad3"]
        }
    }).on('circle-animation-progress', function(event, progress) {
        $(this).find('strong').html(Math.round(10 * progress) + '<i>ft</i>');
    });

    $('#loader_4').circleProgress({
        startAngle: -Math.PI / 4 * 2,
        value: 0.95 ,
        size: 156,
        thickness: 3,
        fill: {
            gradient: ["#189ad3", "#189ad3"]
        }
    }).on('circle-animation-progress', function(event, progress) {
        $(this).find('strong').html(Math.round((Math.random()*20+75) * progress) + '<i>%</i>');
    });

    // Counter
    $(".start-counter").each(function () {
        var counter = $(this);

        counter.countTo({
            formatter: function (value, options) {
                return value.toFixed(options.decimals).replace(/\B(?=(?:\d{3})+(?!\d))/g, ',');
            }
        });
    });

    // Bar Filler
    $('.featured-fund-raised-bar').barfiller({ barColor: '#ff5a00', duration: 1500 });

    $('.fund-raised-bar-1').barfiller({ barColor: '#ff5a00', duration: 1500 });
    $('.fund-raised-bar-2').barfiller({ barColor: '#ff5a00', duration: 1500 });
    $('.fund-raised-bar-3').barfiller({ barColor: '#ff5a00', duration: 1500 });
    $('.fund-raised-bar-4').barfiller({ barColor: '#ff5a00', duration: 1500 });
    $('.fund-raised-bar-5').barfiller({ barColor: '#ff5a00', duration: 1500 });
    $('.fund-raised-bar-6').barfiller({ barColor: '#ff5a00', duration: 1500 });

    // Load more
    let $container      = $('.portfolio-container');
    let $item           = $('.portfolio-item');

    $item.slice(0, 9).addClass('visible');

    $('.load-more-btn').on('click', function (e) {
        e.preventDefault();

        $('.portfolio-item:hidden').slice(0, 9).addClass('visible');
    });



})(jQuery);
