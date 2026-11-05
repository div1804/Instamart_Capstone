
-- Q1: WHERE and HAVING
SELECT
    customer_id,
    SUM(order_amount) AS total_spend
FROM orders
WHERE status = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000;


-- Q2: Three-table JOIN - Revenue by city and category
SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS total_revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;


-- Q3: Scalar subquery - Products above average price
SELECT
    product_id,
    product_name,
    price_inr
FROM products
WHERE price_inr > (
    SELECT AVG(price_inr)
    FROM products
)
ORDER BY price_inr DESC;


-- Q4: LEFT JOIN - Customers without completed orders
SELECT
    c.customer_id,
    c.city
FROM customers c
LEFT JOIN orders o
    ON c.customer_id = o.customer_id
    AND o.status = 'completed'
WHERE o.order_id IS NULL;


-- Q5: CTE - Customers spending above average
WITH customer_spend AS (
    SELECT
        customer_id,
        SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
average_spend AS (
    SELECT AVG(total_spend) AS avg_spend
    FROM customer_spend
)
SELECT
    cs.customer_id,
    ROUND(cs.total_spend, 2) AS total_spend,
    ROUND(a.avg_spend, 2) AS average_spend
FROM customer_spend cs
CROSS JOIN average_spend a
WHERE cs.total_spend > a.avg_spend
ORDER BY cs.total_spend DESC;


-- Q6: Window functions - Customer spending rankings by city
WITH customer_spend AS (
    SELECT
        c.customer_id,
        c.city,
        SUM(o.order_amount) AS total_spend
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT
    customer_id,
    city,
    ROUND(total_spend, 2) AS total_spend,
    RANK() OVER (
        PARTITION BY city ORDER BY total_spend DESC
    ) AS spend_rank,
    DENSE_RANK() OVER (
        PARTITION BY city ORDER BY total_spend DESC
    ) AS dense_rank,
    ROW_NUMBER() OVER (
        PARTITION BY city ORDER BY total_spend DESC
    ) AS row_number
FROM customer_spend
ORDER BY city, spend_rank;


-- Q7: Running total - Daily completed revenue
WITH daily_revenue AS (
    SELECT
        SUBSTR(order_ts, 1, 10) AS day,
        SUM(order_amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT
    day,
    ROUND(revenue, 2) AS daily_revenue,
    ROUND(
        SUM(revenue) OVER (ORDER BY day),
        2
    ) AS cumulative_revenue
FROM daily_revenue
ORDER BY day;
