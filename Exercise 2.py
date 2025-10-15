import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from collections import Counter
import warnings

warnings.filterwarnings('ignore')


def validate_poly_regression(X_train, y_train, X_val, y_val,
                             regressor=None, degrees=range(1, 16),
                             max_features=None):
    """
    Validate polynomial regression models across different degrees.

    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training targets
    X_val : array-like
        Validation features
    y_val : array-like
        Validation targets
    regressor : sklearn estimator, optional
        Regression model to use (default: LinearRegression)
    degrees : iterable
        Polynomial degrees to test
    max_features : int, optional
        Maximum number of features to keep

    Returns:
    --------
    best_model : Pipeline
        Best performing model
    best_rmse : float
        Best RMSE score on validation set
    best_degree : int
        Best polynomial degree
    results : dict
        Dictionary containing all results for analysis
    """

    if regressor is None:
        regressor = LinearRegression()

    # Sample 1% of training data for faster computation
    sample_size = max(int(len(X_train) * 0.01), 100)
    sample_indices = np.random.choice(len(X_train), size=sample_size, replace=False)
    X_train_sample = X_train[sample_indices]
    y_train_sample = y_train[sample_indices]

    print(f"Training on {sample_size} samples ({len(X_train)} total)")
    print(f"Original features: {X_train.shape[1]}")
    print("-" * 70)

    best_rmse = float('inf')
    best_model = None
    best_degree = None
    results = {
        'degrees': [],
        'train_rmse': [],
        'val_rmse': [],
        'n_features': []
    }

    for degree in degrees:
        # Create pipeline
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('regressor', regressor)
        ])

        # Fit model
        try:
            pipeline.fit(X_train_sample, y_train_sample)

            # Get number of features generated
            n_features = pipeline.named_steps['poly'].n_output_features_

            # Make predictions
            y_train_pred = pipeline.predict(X_train_sample)
            y_val_pred = pipeline.predict(X_val)

            # Calculate RMSE
            train_rmse = np.sqrt(mean_squared_error(y_train_sample, y_train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

            # Store results
            results['degrees'].append(degree)
            results['train_rmse'].append(train_rmse)
            results['val_rmse'].append(val_rmse)
            results['n_features'].append(n_features)

            print(f"Degree {degree:2d} | Features: {n_features:6d} | "
                  f"Train RMSE: {train_rmse:8.4f} | Val RMSE: {val_rmse:8.4f}")

            # Update best model
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_model = pipeline
                best_degree = degree

        except Exception as e:
            print(f"Degree {degree:2d} | Failed: {str(e)}")
            continue

    print("-" * 70)
    print(f"Best degree: {best_degree} with validation RMSE: {best_rmse:.4f}")
    print(f"Number of features at best degree: {results['n_features'][best_degree - 1]}")

    return best_model, best_rmse, best_degree, results


def validate_poly_regression_with_regularization(X_train, y_train, X_val, y_val,
                                                 degrees=range(1, 16),
                                                 reg_type='ridge'):
    """
    Validate polynomial regression with regularization (Ridge or Lasso).

    Parameters:
    -----------
    reg_type : str
        'ridge', 'ridgecv', 'lasso', or 'lassocv'
    """

    if reg_type.lower() == 'ridgecv':
        regressor = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000])
    elif reg_type.lower() == 'ridge':
        regressor = Ridge(alpha=1.0)
    elif reg_type.lower() == 'lassocv':
        regressor = LassoCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100], max_iter=5000)
    elif reg_type.lower() == 'lasso':
        regressor = Lasso(alpha=1.0, max_iter=5000)
    else:
        raise ValueError("reg_type must be 'ridge', 'ridgecv', 'lasso', or 'lassocv'")

    print(f"\n{'=' * 70}")
    print(f"Testing with {reg_type.upper()}")
    print(f"{'=' * 70}")

    return validate_poly_regression(X_train, y_train, X_val, y_val,
                                    regressor=regressor, degrees=degrees)


def run_multiple_validations(X_train, y_train, X_val, y_val,
                             n_runs=10, degrees=range(1, 16),
                             regressor=None, reg_name='Linear'):
    """
    Run validation multiple times and analyze distribution of best degrees.

    Parameters:
    -----------
    n_runs : int
        Number of validation runs
    """

    print(f"\n{'#' * 70}")
    print(f"Running {n_runs} validation iterations with {reg_name} Regression")
    print(f"{'#' * 70}\n")

    best_degrees = []
    best_rmses = []
    all_results = []

    for i in range(n_runs):
        print(f"\n{'=' * 70}")
        print(f"RUN {i + 1}/{n_runs}")
        print(f"{'=' * 70}")

        model, rmse, degree, results = validate_poly_regression(
            X_train, y_train, X_val, y_val,
            regressor=regressor, degrees=degrees
        )

        best_degrees.append(degree)
        best_rmses.append(rmse)
        all_results.append(results)

    # Analyze results
    degree_counts = Counter(best_degrees)
    most_common_degree = degree_counts.most_common(1)[0][0]

    print(f"\n{'#' * 70}")
    print("SUMMARY OF ALL RUNS")
    print(f"{'#' * 70}")
    print(f"Best degrees selected: {best_degrees}")
    print(f"Mean RMSE: {np.mean(best_rmses):.4f} (±{np.std(best_rmses):.4f})")
    print(f"Most common degree: {most_common_degree} (selected {degree_counts[most_common_degree]} times)")
    print(f"\nDegree distribution:")
    for degree, count in sorted(degree_counts.items()):
        print(f"  Degree {degree}: {count} times ({count / n_runs * 100:.1f}%)")

    return best_degrees, best_rmses, all_results, most_common_degree


def plot_degree_distribution(best_degrees, reg_name='Linear', save_path=None):
    """
    Plot the distribution of selected polynomial degrees.
    """

    degree_counts = Counter(best_degrees)
    degrees = sorted(degree_counts.keys())
    counts = [degree_counts[d] for d in degrees]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot
    ax1.bar(degrees, counts, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Polynomial Degree', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title(f'Distribution of Best Degrees ({reg_name})',
                  fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xticks(degrees)

    # Add count labels on bars
    for i, (d, c) in enumerate(zip(degrees, counts)):
        ax1.text(d, c + 0.1, str(c), ha='center', va='bottom', fontweight='bold')

    # Histogram/distribution
    ax2.hist(best_degrees, bins=range(min(degrees), max(degrees) + 2),
             color='coral', alpha=0.7, edgecolor='black', align='left')
    ax2.set_xlabel('Polynomial Degree', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title(f'Histogram of Selected Degrees ({reg_name})',
                  fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_validation_curves(results, title='Validation Curves'):
    """
    Plot training and validation RMSE curves.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # RMSE curves
    ax1.plot(results['degrees'], results['train_rmse'],
             'o-', label='Training RMSE', linewidth=2, markersize=8)
    ax1.plot(results['degrees'], results['val_rmse'],
             's-', label='Validation RMSE', linewidth=2, markersize=8)
    ax1.set_xlabel('Polynomial Degree', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_xticks(results['degrees'])

    # Number of features
    ax2.plot(results['degrees'], results['n_features'],
             'o-', color='green', linewidth=2, markersize=8)
    ax2.set_xlabel('Polynomial Degree', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
    ax2.set_title('Feature Count vs Polynomial Degree',
                  fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_xticks(results['degrees'])

    # Add exponential growth annotation
    ax2.text(0.98, 0.98, 'Note: Superlinear growth',
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    X_train = np.random.randn(n_samples, n_features)
    y_train = (X_train[:, 0] ** 2 + 2 * X_train[:, 1] +
               0.5 * X_train[:, 2] * X_train[:, 3] + np.random.randn(n_samples) * 0.5)

    X_val = np.random.randn(200, n_features)
    y_val = (X_val[:, 0] ** 2 + 2 * X_val[:, 1] +
             0.5 * X_val[:, 2] * X_val[:, 3] + np.random.randn(200) * 0.5)

    # 1. Single validation with Linear Regression
    print("\n" + "=" * 70)
    print("SINGLE VALIDATION - LINEAR REGRESSION")
    print("=" * 70)
    model, rmse, degree, results = validate_poly_regression(
        X_train, y_train, X_val, y_val, degrees=range(1, 16)
    )
    plot_validation_curves(results, 'Linear Regression Validation Curves')

    # 2. Test with RidgeCV
    model_ridge, rmse_ridge, degree_ridge, results_ridge = \
        validate_poly_regression_with_regularization(
            X_train, y_train, X_val, y_val, degrees=range(1, 16), reg_type='ridgecv'
        )
    plot_validation_curves(results_ridge, 'RidgeCV Validation Curves')

    # 3. Run 10 times with Linear Regression
    degrees_linear, rmses_linear, all_results_linear, best_deg_linear = \
        run_multiple_validations(X_train, y_train, X_val, y_val,
                                 n_runs=10, degrees=range(1, 16),
                                 regressor=None, reg_name='Linear')
    plot_degree_distribution(degrees_linear, 'Linear Regression')

    # 4. Run 10 times with RidgeCV
    degrees_ridge, rmses_ridge, all_results_ridge, best_deg_ridge = \
        run_multiple_validations(X_train, y_train, X_val, y_val,
                                 n_runs=10, degrees=range(1, 16),
                                 regressor=RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100]),
                                 reg_name='RidgeCV')
    plot_degree_distribution(degrees_ridge, 'RidgeCV Regression')

    print(f"Best degree for Linear Regression: {best_deg_linear}")
    print(f"Best degree for RidgeCV: {best_deg_ridge}")